import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(32)
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')
import blackjax
from blackjax.util import store_only_expectation_values, run_inference_algorithm
from sampler_evaluation.models.student_t import StudentT, Power
import numpy as np
from blackjax.base import SamplingAlgorithm


num_cores = jax.local_device_count()



def thin_algorithm(sampling_algorithm, thinning = 1):

    def step_fn(rng_key, state):
        step = lambda state, rng_key: sampling_algorithm.step(rng_key, state)
        keys = jax.random.split(rng_key, thinning)
        return jax.lax.scan(step, state, keys)

    return SamplingAlgorithm(sampling_algorithm.init, step_fn)


def frobenious(inv_cov):
    """directly convert the covariance matrix expectation values to the bias scalar (to save memory)"""
    
    def func(cov):
        residual = jnp.eye(len(inv_cov)) - inv_cov @ cov
        return jnp.average(jnp.diag(residual @ residual))
    
    return func



def run_mclmc(model, inv_cov, steps, key, 
              desired_energy_variance = 1e-3):

    num_saved_steps = 1000

    init_key, tune_key, run_key = jax.random.split(key, 3)

    initial_position = model.sample_init(init_key)
    initial_state = blackjax.mcmc.mclmc.init(initial_position, model.log_density_fn, init_key)

    integrator = blackjax.mcmc.integrators.isokinetic_velocity_verlet
    
    kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=model.log_density_fn,
        integrator= integrator,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # find values for L and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps= 10000,
        state=initial_state,
        diagonal_preconditioning=False,
        rng_key = tune_key,
        desired_energy_var=desired_energy_variance,
    )

    sampling_alg = blackjax.mclmc(
        model.log_density_fn,
        L= blackjax_mclmc_sampler_params.L,
        step_size= blackjax_mclmc_sampler_params.step_size,
        integrator = integrator
    )

        
    sampling_alg, transform = store_only_expectation_values(
                sampling_algorithm= sampling_alg,
                incremental_value_transform = frobenious(inv_cov),
                state_transform= lambda state: jnp.outer(state.position, state.position), 
                burn_in = 0)
    
    sampling_alg = thin_algorithm(sampling_alg, thinning = steps // num_saved_steps)

    initial_state = sampling_alg.init(blackjax_state_after_tuning)

    bias, info = run_inference_algorithm(
             rng_key=run_key,
             initial_state=initial_state,
             inference_algorithm=sampling_alg,
             num_steps= num_saved_steps,
             transform=transform,
             progress_bar= True,
         )[1]

    eevpd = jnp.std(info.energy_change)**2 / model.ndims
    
    return bias, eevpd



if __name__ == '__main__':

    imodel = int(sys.argv[1])
    Imodel = int(sys.argv[2])

    if Imodel == 0:
        dof = [ 3,  4,  5,  6,  7,  8,  9, 10, 12, 14, 18, 22, 27, 33, 40, 50][imodel]
        model, inv_cov = StudentT(ndims= 100, dof= dof)
    else:
        p = np.arange(2, 10)[imodel]
        model, inv_cov = Power(ndims= 100, p = p)

    key = jax.random.key(0)
    keys= jax.random.split(key, 32)

    
    bias, eevpd = jax.pmap(lambda key: run_mclmc(model, inv_cov, steps=10**7, key = key))(keys)
    
    scratch = '/pscratch/sd/j/jrobnik/bias/'

    jnp.savez(scratch + model.name, bias= bias, eevpd= eevpd)


#shifter --image=jrobnik/sampling:1.0 python3 -m papers.uHMC.identifying_failure