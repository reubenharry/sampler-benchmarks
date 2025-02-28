import functools
import time
import matplotlib.pyplot as plt
# from sampling_algorithms import da_adaptation
import jax
import blackjax
import numpy as np
import jax.numpy as jnp
from blackjax.adaptation.ensemble_mclmc import emaus
from blackjax.mcmc.integrators import mclachlan_coefficients
import jax.scipy.stats as stats
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform, desired_energy_variance= 5e-4):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
        desired_energy_var=desired_energy_variance
    )

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=blackjax_mclmc_sampler_params.L,
        step_size=blackjax_mclmc_sampler_params.step_size,
    )

    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    return samples, blackjax_state_after_tuning, blackjax_mclmc_sampler_params, run_key


im = 0 + 1j

def run_emaus(
        sample_init,
        logdensity_fn,
        ndims,
        transform,
        key,
        diagonal_preconditioning,
    ):
        mesh = jax.sharding.Mesh(jax.devices(), "chains")

        integrator_coefficients = mclachlan_coefficients

        info, grads_per_step, _acc_prob, final_state = emaus(
            logdensity_fn=logdensity_fn,
            sample_init=sample_init,
            transform=transform,
            ndims=ndims,
            num_steps1=500,
            num_steps2=500,
            num_chains=5000,
            mesh=mesh,
            rng_key=key,
            alpha=1.9,
            C=0.1,
            early_stop=0,
            r_end=1e-2,
            diagonal_preconditioning=diagonal_preconditioning,
            integrator_coefficients=integrator_coefficients,
            steps_per_sample=15,
            acc_prob=None,
            ensemble_observables=lambda x: x,
            # ensemble_observables = lambda x: vec @ x
        )  # run the algorithm

        print((info["phase_2"][1].shape), "SHAPE")

        # output = info["phase_2"][1][:, :, :]

        # print((output.shape), "output SHAPE")

        # output =  output.reshape(output.shape[0] * output.shape[1], output.shape[2])

        # return output

        return final_state.position

def mod_index(arr, i):
    return arr[i % (arr.shape[0])]

def sample_s_chi(U, r, t=1, i=1, beta=1, hbar=1, m =1, num_steps=100000, rng_key=jax.random.PRNGKey(0)):


    P = r.shape[0] - 1

    sqnorm = lambda x: x.dot(x)

    def make_M_Minv_K(P, t):
        tau_c = t - ((beta * hbar * im) / 2)

        alpha = (m*P*beta)/(4*(jnp.abs(tau_c)**2))
        gamma = (m*P*t)/(hbar * (jnp.abs(tau_c)**2)) 

        M = (jnp.diag(2*alpha  + (beta / (4*P))*jax.vmap(jax.grad(jax.grad(U)))(r[1:-1])) ) - alpha * jnp.diag(jnp.ones(P-2),k=1) - alpha * jnp.diag(jnp.ones(P-2),k=-1)
        # M = jnp.diag(jnp.ones((P-1,)))

        Minv = jnp.linalg.inv(M)

        K = gamma * (2*r[1:-1] - r[:-2] - r[2:]) - (t * jax.vmap(jax.grad(U))(r[1:-1]))/(P*hbar)

        print(K.shape, "k shape")

        return M, Minv, K, alpha, gamma, r
    
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t)

    @jax.jit
    def logdensity_fn(s):
        term1 = (alpha / 2) * (sqnorm(s[1:] - s[:-1]) + (  (s[0]**2) + (s[-1]**2) ))
        term2 = (beta / (2*P)) * jnp.sum(jax.vmap(U)(r[1:-1] + s/2) + jax.vmap(U)(r[1:-1] - s/2))
        return  -(term1 + term2)

    def xi(s):
        term1 = gamma * ((r[2:-1] - r[1:-2]).dot(s[1:] - s[:-1])  + (r[1] - r[0])*s[0] + (r[-1] - r[-2])*s[-1] )
        term2 = -(t/(P*hbar))*jnp.sum(jax.vmap(U)(r[1:-1] + s/2) - jax.vmap(U)(r[1:-1] - s/2)  )
        return term1 + term2
    
    def transform(state, info):
        x = state.position
        return (xi(x),x[i])
        
    
    init_key, run_key = jax.random.split(rng_key)
    
    toc = time.time()

    sequential = True
    if sequential:

        # model = undefined

        from collections import namedtuple
        Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])
        
        model = Model(
            ndims=(P-1),
            log_density_fn=logdensity_fn,
            default_event_space_bijector=lambda x: x,
        )

        samples, metadata = unadjusted_mclmc(return_samples=True)(
            model=model, 
            num_steps=50000,
            initial_position=jax.random.normal(init_key, (P-1,)), 
            key=jax.random.key(0))

        plt.hist(samples[:, i], bins=100, density=True)
        plt.savefig("nuts.png")

        # print(jnp.expand_dims(samples[:, i],[0,2]).shape, "baz")
        # print(jax.vmap(xi)(samples).shape)

        # raise Exception

        print((samples[:, i]**2).mean())
        print(K @ Minv @ K, "KMK")

        error_at_each_step = get_standardized_squared_error(
            jnp.expand_dims(jax.vmap(xi)(samples),[0,2]), 
            f=lambda x: x**2,
            E_f=(K @ Minv @ K),
            Var_f=2.0 * (K @ Minv @ K)**2,
        )

        # print(error_at_each_step[-10:], "error at each step")

        # gradient_calls_per_chain = metadata['num_grads_per_proposal'].mean()
        gradient_calls_per_chain = 2.0 

        print(samples_to_low_error(error_at_each_step, low_error=1/100) * gradient_calls_per_chain)
       
        # raise Exception
       
       
        # (samples, weights), initial_state, params, chain_key = run_nuts(
        #         logdensity_fn=logdensity_fn,
        #         num_steps=num_steps,
        #         initial_position=jax.random.normal(init_key, (P-1,)),
        #         key=run_key,
        #         transform=transform,
        #         # desired_energy_variance=0.0005
        #     )
        # print(samples.shape, "sshape")
        # samples, weights = samples[::3], weights[::3]
        # raise Exception



        (samples, weights), initial_state, params, chain_key = run_mclmc(
                logdensity_fn=logdensity_fn,
                num_steps=num_steps,
                initial_position=jax.random.normal(init_key, (P-1,)),
                key=run_key,
                transform=transform,
                # desired_energy_variance=0.0005
            )
        

    else:

        sample_init = lambda init_key: jax.random.normal(key=init_key, shape=(P-1,))

        
        # sample_init = lambda key: jax.random.normal(key, shape=(2,)) * jnp.array([10.0, 5.0]) * 2
        # sample_init = lambda key: jax.random.normal(key, shape=(3,))

        # def logdensity_fn(x):
        #      return 1.0

        # def logdensity_fn(x):
        #     mu2 = 0.03 * (x[0] ** 2 - 100)
        #     return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

        raw_samples = run_emaus(
            sample_init=sample_init,
            logdensity_fn=logdensity_fn,
            transform=lambda x: x,
            # transform=lambda x : jnp.array([xi(x), x[i]]),
            ndims=P-1,
            # ndims=2,
            key=jax.random.key(0),
            diagonal_preconditioning=False,
        )

        print(raw_samples.shape, "sample shape")

        error_at_each_step = get_standardized_squared_error(
            jnp.expand_dims(jax.vmap(xi)(raw_samples),[0,2]), 
            f=lambda x: x**2,
            E_f=(K @ Minv @ K),
            Var_f=2.0 * (K @ Minv @ K)**2,
        )

        # print(error_at_each_step[-10:], "error at each step")

        # gradient_calls_per_chain = metadata['num_grads_per_proposal'].mean()
        gradient_calls_per_chain = 2.0 

        print(samples_to_low_error(error_at_each_step, low_error=1/100) * gradient_calls_per_chain)
       
        # raise Exception

        samples, weights = (jax.vmap(lambda x : (xi(x), x[i]))(raw_samples))
        # samples, weights = raw_samples # (jax.vmap(lambda x : (xi(x), x[i]))(raw_samples))
        # raise Exception
    tic = time.time()
    print(tic - toc, "time")
    
    
    def analytic_gaussian(l):
        return jax.scipy.stats.norm.pdf(loc=0, scale=jnp.sqrt(K @ Minv @ K), x=l)
    
    def analytic(lam, i): 

        return (1/ (2*np.sqrt(2*np.pi))) *2*(Minv[:,i]@K)*((1/(K @ Minv @ K))**(3/2))*lam*np.exp( (-(lam**2)) / (2 * K @ Minv @ K) )

    num_bins = 100

    samples = np.array(samples)
    weights = np.array(weights)
    hist, edges = np.histogram(samples, bins=num_bins)
    normalization_constant = np.sum((edges[1:] - edges[:-1])*hist)
    plt.hist(samples, bins=num_bins, density=False, weights=weights/normalization_constant)

    l = np.linspace(jnp.min(samples), jnp.max(samples), num_bins)
    solution = analytic(lam=l, i = i,)
    plt.plot(l, solution)

    plt.savefig("mclmc_expectation.png")

    # new plot
    plt.clf()

    gaussian = analytic_gaussian(l)
    plt.plot(l, gaussian)
    plt.hist(samples, bins=num_bins, density=True)

    plt.savefig("mclmc_normal.png")

    # samples are \xi(s), weights are s[i]
    return samples, weights



beta_hbar_omega = 15.8
m_omega_over_hbar = 0.03
m = 1.0
hbar = 1.0
omega = (m_omega_over_hbar * hbar) / m
beta = (beta_hbar_omega / (hbar * omega))

r_length = 33

if __name__ == "__main__":

    # def regression_logprob(log_scale, coefs, preds, x):
    #     """Linear regression"""
    #     scale = jnp.exp(log_scale)
    #     scale_prior = stats.expon.logpdf(scale, 0, 1) + log_scale
    #     coefs_prior = stats.norm.logpdf(coefs, 0, 5)
    #     y = jnp.dot(x, coefs)
    #     logpdf = stats.norm.logpdf(preds, y, scale)
    #     # reduce sum otherwise broacasting will make the logprob biased.
    #     return sum(x.sum() for x in [scale_prior, coefs_prior, logpdf])

    # init_key0, init_key1, inference_key = jax.random.split(jax.random.key(0), 3)

    # x_data = jax.random.normal(init_key0, shape=(1000, 1))
    # y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

    # logposterior_fn_ = functools.partial(
    #     regression_logprob, x=x_data, preds=y_data
    # )
     
    # logdensity_fn = lambda x: logposterior_fn_(
    #         coefs=x["coefs"][0], log_scale=x["log_scale"][0]
    #     )

    # def sample_init(key):
    #     key1, key2 = jax.random.split(key)
    #     coefs = jax.random.uniform(key1, shape=(1,), minval=1, maxval=2)
    #     log_scale = jax.random.uniform(key2, shape=(1,), minval=1, maxval=2)
    #     return {"coefs": coefs, "log_scale": log_scale}

    
    
    samples, weights = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        # r=jax.random.uniform(jax.random.PRNGKey(1), (r_length,)),
        num_steps=50000,
        )
    
