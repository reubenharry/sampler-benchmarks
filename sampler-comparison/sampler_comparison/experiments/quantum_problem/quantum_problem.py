import functools
import time
import matplotlib.pyplot as plt
# from sampling_algorithms import da_adaptation
import sys
sys.path.append("/global/u1/r/reubenh/blackjax")
sys.path.append("/global/u1/r/reubenh/sampler-benchmarks/sampler-comparison")
import jax
# import blackjax
import numpy as np
import jax.numpy as jnp
from blackjax.adaptation.ensemble_mclmc import laps
from blackjax.mcmc.integrators import mclachlan_coefficients
import jax.scipy.stats as stats
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
    unadjusted_mclmc_no_tuning,
)
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from blackjax.diagnostics import effective_sample_size

im = 0 + 1j

def run_laps(
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

        info, grads_per_step, _acc_prob, final_state = laps(
    
            logdensity_fn=logdensity_fn, 
            sample_init= sample_init,
            ndims=ndims, 
            num_steps1=num_unadjusted_steps, 
            num_steps2=num_adjusted_steps, 
            num_chains=num_chains, 
            mesh=mesh, 
            rng_key=key, 
            early_stop=True,
            diagonal_preconditioning=diagonal_preconditioning, 
            integrator_coefficients= integrator_coefficients, 
            steps_per_sample=15,
            ensemble_observables= lambda x: x,
            observables_for_bias=lambda x: jnp.square(x),
            contract = lambda _: jnp.array([0.0, 0.0]),
            r_end=0.01,
            diagnostics= True,
            superchain_size= 1
            ) 
        
        #  info, grads_per_step, _acc_prob, final_state = laps(
        #     logdensity_fn=logdensity_fn,
        #     sample_init=sample_init,
        #     # transform=transform,
        #     ndims=ndims,
        #     num_steps1=num_unadjusted_steps,
        #     num_steps2=num_adjusted_steps,
        #     num_chains=num_chains,
        #     mesh=mesh,
        #     rng_key=key,
        #     alpha=1.9,
        #     C=0.1,
        #     early_stop=True,
        #     r_end=1e-2,
        #     diagonal_preconditioning=diagonal_preconditioning,
        #     integrator_coefficients=integrator_coefficients,
        #     steps_per_sample=15,
        #     acc_prob=None,
        #     ensemble_observables=lambda x: x,
        #     observables_for_bias=lambda x: jnp.square(x),
        #     # ensemble_observables = lambda x: vec @ x
        # )  # run the algorithm


        print((info["phase_2"][1].shape), "SHAPE")

        print("steps_done", info["phase_1"]["steps_done"])

        return final_state.position

        results = info["phase_2"][1]

        reshaped_results = results.reshape(results.shape[0]*results.shape[1], results.shape[2])

        print(reshaped_results.shape, "reshaped results shape")

        return reshaped_results


        return results.reshape(results.shape[0]*results.shape[1], results.shape[2])

        return final_state.position

def mod_index(arr, i):
    return arr[i % (arr.shape[0])]

def analytic_gaussian(l, K, Minv):
    return jax.scipy.stats.norm.pdf(loc=0, scale=jnp.sqrt(K @ Minv @ K), x=l)

def analytic(lam, i, K, Minv): 

    return (1/ (2*np.sqrt(2*np.pi))) *2*(Minv[:,i]@K)*((1/(K @ Minv @ K))**(3/2))*lam*np.exp( (-(lam**2)) / (2 * K @ Minv @ K) )


def make_histograms(filename, samples, weights, K, Minv, i):

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

def xi(s, r, U, t, P, hbar, gamma):
    term1 = gamma * ((r[2:-1] - r[1:-2]).dot(s[1:] - s[:-1])  + (r[1] - r[0])*s[0] - (r[-1] - r[-2])*s[-1] )
    # term1 = gamma * (jnp.sum(jnp.array([((r[i] - r[i-1])*(s[i-1] - s[i-2])) for i in range(2, P)])) + (r[1] - r[0])*s[0] - (r[P] - r[P-1])*s[P-2] )
    # print(term1, "term1", gamma)
    # term2 = -(t/(P*hbar))*jnp.sum(jax.vmap(U)(r[1:-1] + s/2) - jax.vmap(U)(r[1:-1] - s/2)  )
    term2 = -(t/(P*hbar))*jnp.sum(jnp.array([U(r[i-1]+(s[i-2]/2)) - U(r[i-1]-(s[i-2]/2)) for i in range (2, P+1)]))
    return term1 + term2




def sample_s_chi(U, r, t=1, i=1, beta=1, hbar=1, m =1, rng_key=jax.random.PRNGKey(0), sequential=False, sample_init=None, num_unadjusted_steps=100, num_adjusted_steps=100, num_chains=5000, initial_ss_and_params=None):


    P = r.shape[0] - 1
    print(P, "P")

    sqnorm = lambda x: x.dot(x)

    
    M, Minv, K, alpha, gamma, r = make_M_Minv_K(P, t, U, r, beta, hbar, m)

    @jax.jit
    def logdensity_fn(s):
        term1 = (alpha / 2) * (sqnorm(s[1:] - s[:-1]) + (  (s[0]**2) + (s[-1]**2) ))
        term2 = (beta / (2*P)) * jnp.sum(jax.vmap(U)(r[1:-1] + s/2) + jax.vmap(U)(r[1:-1] - s/2))
        return  -(term1 + term2)
    


    
    def transform(state, info):
        x = state.position
        return (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma
                   
                   ),x[i])
        
    
    init_key, run_key = jax.random.split(rng_key)
    
    toc = time.time()

    # print(logdensity_fn(jnp.ones((P-1,))), "logdensity"
    # )
    # print(xi(jnp.ones((P-1,)),r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), "xi(1)")
    # print(xi(-jnp.ones((P-1,)),r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), "xi(-1)")
    # print("shapes", P, r.shape)
    # raise Exception("Stop here")

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

        raw_samples = raw_samples[1000000:, :]
        # save raw samples
        # np.save(filename + ".npy", raw_samples)
        
        # raw_samples, metadata = unadjusted_lmc(return_samples=True)(
        #     model=model, 
        #     num_steps=num_unadjusted_steps,
        #     initial_position=jax.random.normal(init_key, (P-1,)), 
        #     key=jax.random.key(0))

        samples, weights = (jax.vmap(lambda x : (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples))

    else:

        # if previous_samples is None:

        #     sample_init = lambda init_key: jax.random.normal(key=init_key, shape=(P-1,))
        # else:
        #     sample_init = lambda init_key: previous_samples

        laps = False
        if laps:
            tic = time.time()
            raw_samples = run_laps(
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

            print(time.time() - tic, "time")

        else:

            from collections import namedtuple
            Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])
            
            model = Model(
                ndims=(P-1),
                log_density_fn=logdensity_fn,
                default_event_space_bijector=lambda x: x,
            )

#             [12.668048 20.329592 25.034225 27.92579  29.679482 30.729008 31.375006                                                
#  31.75311  32.045685 32.186428 32.27571  32.31823  32.346756 32.355663
#  32.393826 32.406498 32.42128  32.433342 32.42514  32.40407  32.370388
#  32.26124  32.066486 31.854162 31.436607 30.826267 29.737642 27.963968
#  25.037106 20.369642 12.662694]

            import blackjax
            # loaded_samples = np.load("sequential_umclmc" + ".npy")
            

            # initial_state = blackjax.mclmc.init(
            #     position=jax.random.normal(jax.random.key(0), (P-1,)),
            #     logdensity_fn=logdensity_fn,
            #     random_generator_arg=jax.random.key(0),
            # )

            

            tic = time.time()
            if initial_ss_and_params is not None:

                raw_samples, metadata = jax.vmap(lambda key, pos: 
                    unadjusted_mclmc_no_tuning(
                        return_samples=True, 
                        # return_only_final=True, 
                        integrator_type="mclachlan",
                        L=initial_ss_and_params['params']['L'], 
                        step_size=initial_ss_and_params['params']['step_size'], 
                        initial_state=blackjax.mclmc.init(
                            position=pos,
                            logdensity_fn=logdensity_fn,
                            random_generator_arg=jax.random.split(key)[0],
                        ), 
                        inverse_mass_matrix=initial_ss_and_params['params']['inverse_mass_matrix'],
                    )(
                    model=model, 
                    num_steps=num_unadjusted_steps,
                    initial_position=pos, 
                    key=key))(jax.random.split(init_key, num_chains), initial_ss_and_params['ss'])

                # raw_samples = raw_samples.reshape(num_chains*num_unadjusted_steps, P-1)
                samples, weights = (jax.vmap(lambda x : (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples.reshape(num_chains*num_unadjusted_steps, P-1)))
                return samples, weights, raw_samples, K, Minv

            else:
                raw_samples, metadata = jax.vmap(lambda key: unadjusted_mclmc(
                    return_samples=True, 
                    return_only_final=False,
                    num_tuning_steps=1000,
                    )(
                        model=model, 
                        num_steps=num_unadjusted_steps,
                        
                        initial_position=jax.random.normal(init_key, (P-1,)), 
                        key=key))(jax.random.split(init_key, num_chains))
                jax.debug.print("metadata {x}", x=metadata.keys())
                # raise Exception("Stop here")

                L = metadata['L'].mean()
                step_size = metadata['step_size'].mean()
                inverse_mass_matrix = metadata['inverse_mass_matrix'].mean(axis=0)

                samples = raw_samples

                # samples, weights = (jax.vmap(lambda x : (xi(x,r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma), x[i]))(raw_samples))
                return samples, L, step_size, inverse_mass_matrix

            # print(time.time() - tic, "time (parallel, no laps)")
            # raw_samples = raw_samples.reshape(num_chains*num_unadjusted_steps, P-1)
            # raise Exception("Stop here")
            # print(metadata['inverse_mass_matrix'].mean(axis=0), "metadata")

        # print(raw_samples.shape, "raw samples shape")
        # raise Exception("Stop here")

            # raw_samples = raw_samples[:, -1, :]

            # print(raw_samples.shape, "raw samples shape")
            # raise Exception("Stop here")



        # print(raw_samples.shape, "sample shape")



    # error_at_each_step = get_standardized_squared_error(
    #     jnp.expand_dims(jax.vmap(functools.partial(xi, r=r,U=U,t=t,P=P,hbar=hbar,gamma=gamma))(raw_samples),[0,2]), 
    #     f=lambda x: x**2,
    #     E_f=(K @ Minv @ K),
    #     Var_f=2.0 * (K @ Minv @ K)**2,
    # )
    # gradient_calls_per_chain = metadata['num_grads_per_proposal'].mean()
    # gradient_calls_per_chain = 2.0 
    # print(samples_to_low_error(error_at_each_step, low_error=1/100), "ess")
    
    # tic = time.time()
    # print(tic - toc, "time")
    
    
        
        # np.save(filename + ".npy", raw_samples)
    
    # samples are \xi(s), weights are s[i]




if __name__ == "__main__":

    """

    """

    # P = 32, m = hbar = omega = 1, beta = 10 and r = [ -2.2    -2.005625  -1.53125  -1.196875  -1.5625   -1.328125  -1.19375  -1.259375  -1.125   -0.990625  -0.65625  -0.821875   -0.5875   -0.4453125  -0.32421875  -0.184375  -0.16346   0.0084375   0.121875  0.5353125  0.4875   0.5621875  0.75625   0.890625   1.1125   1.2259375  1.339375  1.4428124  1.55625   1.6696875   1.783125  1.8965625  2.1 ]

    beta_hbar_omega = 15.8
    m_omega_over_hbar = 0.03
    m = 1.0
    hbar = 1.0
    omega = (m_omega_over_hbar * hbar) / m
    beta = (beta_hbar_omega / (hbar * omega))

    r_length = 33
    P = r_length - 1

    t=1
    i=1


    # r=jax.random.normal(jax.random.PRNGKey(3), (r_length,))
    # r = jnp.array([ -2.2,    -2.005625,  -1.53125,  -1.196875,  -1.5625,   -1.328125,  -1.19375,  -1.259375,  -1.125,   -0.990625,  -0.65625,  -0.821875,   -0.5875,   -0.4453125,  -0.32421875,  -0.184375,  -0.16346,   0.0084375,   0.121875,  0.5353125,  0.4875,   0.5621875,  0.75625 ,  0.890625 ,  1.1125 ,  1.2259375 , 1.339375 , 1.4428124 , 1.55625  , 1.6696875,   1.783125 , 1.8965625 , 2.1 ])    


    r = jnp.array([ -1.2,    2.005625,  -0.8,  -1.196875,  -1.5625,   -1.328125,  -1.19375,  -1.259375,  -1.125,   -0.990625,  -0.65625,  -0.821875,   -0.5875,   -0.4453125,  -0.32421875,  -0.184375,  -0.16346,   0.0084375,   0.121875,  0.5353125,  0.4875,   0.5621875,  0.75625 ,  0.890625 ,  1.1125 ,  1.2259375 , 1.339375 , 1.4428124 , 1.55625  , 1.6696875,   1.783125 , 1.8965625 , 2.1 ])    
    

    ## This is the sequential version of the code (only runs a single chain)
    ## you shouldn't need to use it, but it's here for reference
    ## Note that it may be a fair bit faster (in the sense of requiring fewer samples) than the last version, because I added preconditioning
    # samples, weights, raw_samples = sample_s_chi(
    #     t=1,
    #     i=1,
    #     beta=beta,
    #     hbar=hbar,
    #     m=m,
    #     U = lambda x : 0.5*m*(omega**2)*(x**2),
    #     r=r,
    #     # r=jax.random.uniform(jax.random.PRNGKey(1), (r_length,)),
    #     sequential=True,
    #     sample_init=lambda init_key: jax.random.normal(key=init_key, shape=(r_length-2,)),
    #     num_unadjusted_steps=2000000,
    #     # num_adjusted_steps=5000,
    #     filename="sequential_umclmc",
    #     # filename="nuts",
    #     )
    
    # ess = effective_sample_size(samples[None,:,None])

    # print(ess, "ess")
    # raise Exception("Stop here")

    
    



    # raise Exception("Stop here")

    # load raw samples
    filename = "parallel_umclmc"
    raw_samples = np.load(filename + ".npy")
    # sample_init = lambda init_key: jax.random.choice(init_key, raw_samples, axis=0)
    sample_init=lambda init_key: jax.random.normal(key=init_key, shape=(r_length-2,))



    # This is the parallel version of the code, to be run on the first iteration of the inner loop.
    # samples, weights, raw_samples = sample_s_chi(
    #     t=1,
    #     i=1,
    #     beta=beta,
    #     hbar=hbar,
    #     m=m,
    #     U = lambda x : 0.5*m*(omega**2)*(x**2),
    #     r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
    #     sequential=False,

    #     sample_init=sample_init,
    #     num_unadjusted_steps=1, # md = unadjusted. This could probably be fewer.
    #     num_adjusted_steps=10, # mc = adjusted. 
    #     filename=filename,
    #     num_chains=1000 # this should be at large as possible while still fitting in memory.
    #     )

    num_chains = 10000

    samples, L, step_size, inverse_mass_matrix = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        sequential=False,
        initial_ss_and_params=None,
        sample_init=sample_init,
        num_unadjusted_steps=2000, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        # filename=filename,
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )

    # params = {'L' : 54.682884, 'step_size' : 2.9930937, 'inverse_mass_matrix' : jnp.array([12.668048, 20.329592, 25.034225, 27.92579, 29.679482, 30.729008, 31.375006, 31.75311, 32.045685, 32.186428, 32.27571, 32.31823, 32.346756, 32.355663, 32.393826, 32.406498, 32.42128, 32.433342, 32.42514, 32.40407, 32.370388, 32.26124, 32.066486, 31.854162, 31.436607, 30.826267, 29.737642, 27.963968, 25.037106, 20.369642, 12.662694])}
    params = {'L' : L, 'step_size' : step_size, 'inverse_mass_matrix' : inverse_mass_matrix}
    # initial_ss_and_params = {'params': params, 'ss': samples[:, -1, :]}
    loaded_samples = jax.random.normal(jax.random.key(0), (num_chains, P-1))
    initial_ss_and_params = {'params': params, 'ss': loaded_samples}
    
    tic = time.time()
    samples, weights, raw_samples, K, Minv = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        sequential=False,
        initial_ss_and_params=initial_ss_and_params,
        sample_init=sample_init,
        num_unadjusted_steps=100, # md = unadjusted. This could probably be fewer.
        num_adjusted_steps=0, # mc = adjusted. 
        # filename=filename,
        num_chains=num_chains # this should be at large as possible while still fitting in memory.
        )
    print(time.time() - tic, "time")
    if filename:   
        print("making histograms")
        make_histograms(filename, samples=samples, weights=weights, K=K, Minv=Minv, i=i)
        # save samples

    print(samples.shape, "samples shape")
    # effective sample size
    ess = effective_sample_size(samples[None,:,None])

    print(ess, "ess")

    raise Exception("Stop here")
    
    
    ### This is the parallel version of the code, to be run on the subsequent iterations of the inner loop. Note that `sample_init` depends on the raw samples (i.e. the P-dimensional samples) from the previous inner loop calculation.
    sample_init = lambda init_key: jax.random.choice(init_key, raw_samples, axis=0)
    samples, weights, raw_samples = sample_s_chi(
        t=1,
        i=1,
        beta=beta,
        hbar=hbar,
        m=m,
        U = lambda x : 0.5*m*(omega**2)*(x**2),
        r=jax.random.normal(jax.random.PRNGKey(3), (r_length,)),
        sequential=False,
        sample_init=sample_init,
        num_unadjusted_steps=100, # this is intentionally chosen to be small. For large outer steps, it could be made larger. Conveniently, the burn_in phase will abort early if it detects convergence (across chains), so there's often litle harm in making this number larger. 
        num_adjusted_steps=50, # it's likely that this could also be smaller, especially in the regime where you only need a single effective sample per chain.
        # filename="second",
        num_chains=12000
        )
    

