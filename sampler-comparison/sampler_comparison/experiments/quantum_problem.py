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


im = 0 + 1j

def run_emaus(
        sample_init,
        logdensity_fn,
        ndims,
        transform,
        key,
        diagonal_preconditioning,
        num_unadjusted_steps,
        num_adjusted_steps,
        num_chains
    ):
        mesh = jax.sharding.Mesh(jax.devices(), "chains")

        integrator_coefficients = mclachlan_coefficients

        info, grads_per_step, _acc_prob, final_state = emaus(
            logdensity_fn=logdensity_fn,
            sample_init=sample_init,
            transform=transform,
            ndims=ndims,
            num_steps1=num_unadjusted_steps,
            num_steps2=num_adjusted_steps,
            num_chains=num_chains,
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

        # print((info["phase_2"][1].shape), "SHAPE")

        return final_state.position

def mod_index(arr, i):
    return arr[i % (arr.shape[0])]

def analytic_gaussian(l, K, Minv):
    return jax.scipy.stats.norm.pdf(loc=0, scale=jnp.sqrt(K @ Minv @ K), x=l)

def analytic(lam, i, K, Minv): 

    return (1/ (2*np.sqrt(2*np.pi))) *2*(Minv[:,i]@K)*((1/(K @ Minv @ K))**(3/2))*lam*np.exp( (-(lam**2)) / (2 * K @ Minv @ K) )


def make_unsigned_histogram(filename, samples, weights, K, Minv, i):

    num_bins = 100
    samples = np.array(samples)
    weights = np.array(weights)
    hist, edges = np.histogram(samples, bins=num_bins)
    normalization_constant = np.sum((edges[1:] - edges[:-1])*hist)
    plt.hist(samples, bins=num_bins, density=False, weights=weights/normalization_constant)
    l = np.linspace(jnp.min(samples), jnp.max(samples), num_bins)
    solution = analytic(lam=l, i = i, K=K, Minv=Minv)
    plt.plot(l, solution)
    plt.savefig(filename)
    plt.clf()

    gaussian = analytic_gaussian(l, K=K, Minv=Minv)
    plt.plot(l, gaussian)
    plt.hist(samples, bins=num_bins, density=True)

    plt.savefig(filename + "hist.png")
    plt.clf()

def make_M_Minv_K(P, t, U, r, beta, hbar,m):
    tau_c = t - ((beta * hbar * im) / 2)

    alpha = (m*P*beta)/(4*(jnp.abs(tau_c)**2))
    gamma = (m*P*t)/(hbar * (jnp.abs(tau_c)**2)) 

    M = (jnp.diag(2*alpha  + (beta / (4*P))*jax.vmap(jax.grad(jax.grad(U)))(r[1:-1])) ) - alpha * jnp.diag(jnp.ones(P-2),k=1) - alpha * jnp.diag(jnp.ones(P-2),k=-1)

    Minv = jnp.linalg.inv(M)

    K = gamma * (2*r[1:-1] - r[:-2] - r[2:]) - (t * jax.vmap(jax.grad(U))(r[1:-1]))/(P*hbar)

    print(K.shape, "k shape")

    return M, Minv, K, alpha, gamma, r


def sample_s_chi(U, r, t=1, i=1, beta=1, hbar=1, m =1, rng_key=jax.random.PRNGKey(0), sequential=False, sample_init=None, num_unadjusted_steps=100, num_adjusted_steps=100, num_chains=5000, filename=""):


    P = r.shape[0] - 1

    sqnorm = lambda x: x.dot(x)

    
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar, m)

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

    if sequential:

        from collections import namedtuple
        Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])
        
        model = Model(
            ndims=(P-1),
            log_density_fn=logdensity_fn,
            default_event_space_bijector=lambda x: x,
        )

        raw_samples, metadata = unadjusted_mclmc(return_samples=True)(
            model=model, 
            num_steps=num_unadjusted_steps,
            initial_position=jax.random.normal(init_key, (P-1,)), 
            key=jax.random.key(0))

        samples, weights = (jax.vmap(lambda x : (xi(x), x[i]))(raw_samples))

    else:

        # if previous_samples is None:

        #     sample_init = lambda init_key: jax.random.normal(key=init_key, shape=(P-1,))
        # else:
        #     sample_init = lambda init_key: previous_samples

        raw_samples = run_emaus(
            sample_init=sample_init,
            logdensity_fn=logdensity_fn,
            transform=lambda x: x,
            ndims=P-1,
            key=jax.random.key(0),
            diagonal_preconditioning=True,
            num_adjusted_steps=num_adjusted_steps,
            num_unadjusted_steps=num_unadjusted_steps,
            num_chains=num_chains
        )

        print(raw_samples.shape, "sample shape")

        error_at_each_step = get_standardized_squared_error(
            jnp.expand_dims(jax.vmap(xi)(raw_samples),[0,2]), 
            f=lambda x: x**2,
            E_f=(K @ Minv @ K),
            Var_f=2.0 * (K @ Minv @ K)**2,
        )


        # gradient_calls_per_chain = metadata['num_grads_per_proposal'].mean()
        # gradient_calls_per_chain = 2.0 
        # print(samples_to_low_error(error_at_each_step, low_error=1/100) * gradient_calls_per_chain)

        samples, weights = (jax.vmap(lambda x : (xi(x), x[i]))(raw_samples))
    
    tic = time.time()
    print(tic - toc, "time")
    
    make_unsigned_histogram(filename, samples=samples, weights=weights, K=K, Minv=Minv, i=i)
    
    # samples are \xi(s), weights are s[i]
    return samples, weights, raw_samples



beta_hbar_omega = 15.8
m_omega_over_hbar = 0.03
m = 1.0
hbar = 1.0
omega = (m_omega_over_hbar * hbar) / m
beta = (beta_hbar_omega / (hbar * omega))

r_length = 33

if __name__ == "__main__":


    samples, weights, raw_samples = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        # r=jax.random.uniform(jax.random.PRNGKey(1), (r_length,)),
        sequential=True,
        sample_init=lambda init_key: jax.random.normal(key=init_key, shape=(r_length-2,)),
        num_unadjusted_steps=50000,
        num_adjusted_steps=0,
        filename="first",
        # num_chains=5000
        )
    
    
    samples, weights, raw_samples = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        # r=jax.random.uniform(jax.random.PRNGKey(1), (r_length,)),
        sequential=False,
        sample_init=lambda init_key: jax.random.normal(key=init_key, shape=(r_length-2,)),
        num_unadjusted_steps=500,
        num_adjusted_steps=500,
        filename="second",
        num_chains=15000
        )
    
    

    # sample_init = lambda init_key: jax.random.choice(init_key, raw_samples, axis=0)



    
    # samples, weights, raw_samples = sample_s_chi(
    #     t=1,
    #     i=1,
    #     beta=beta,
    #     hbar=hbar,
    #     m=m,
    #     U = lambda x : 0.5*m*(omega**2)*(x**2),
    #     r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
    #     # r=jax.random.uniform(jax.random.PRNGKey(1), (r_length,)),
    #     sequential=False,
    #     sample_init=sample_init,
    #     num_unadjusted_steps=10,
    #     num_adjusted_steps=10,
    #     filename="second",
    #     num_chains=5000
    #     )
    


#     jax.debug.print("foo {x}",x=raw_samples.shape)

#     ixs = jax.vmap(lambda k: jax.random.choice(key=k,a=jax.numpy.arange(50000)))(jax.random.split(jax.random.key(0),50000))

#     new_samples = samples[ixs]
#     new_weights = weights[ixs]

#     M, Minv, K, alpha, gamma, r = make_M_Minv_K(32, 1, U=lambda x : 0.5*m*(omega**2)*(x**2), r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)), beta=beta, hbar=hbar, m=m)
#     make_unsigned_histogram("third", new_samples, new_weights, K, Minv, 1)

#     # jax.debug.print("bar {x}", x=jax.vmap(lambda k: jax.random.choice(k, raw_samples, axis=0,))(jax.random.split(jax.random.key(0), 50000)).shape)