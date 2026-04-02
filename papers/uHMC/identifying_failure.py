import os, sys
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
sys.path.append('../blackjax/')
sys.path.append('sampler-evaluation/')
sys.path.append('sampler-comparison/')
import blackjax
from blackjax.util import thin_algorithm, store_only_expectation_values, run_inference_algorithm
from sampler_evaluation.models.student_t import StudentT

num_cores = jax.local_device_count()
print(num_cores)


def frobenious(model):
    """directly convert the covariance matrix expectation values to the bias scalar (to save memory)"""
    
    def func(cov):
        residual = jnp.eye(model.ndims) - model.inv_cov @ cov
        return jnp.average(jnp.diag(residual @ residual))
    
    return func



def run_mclmc(model, step_size, L, steps, key):

    num_saved_steps = 1000

    init_key, run_key = jax.random.split(key, 2)

    initial_position = model.sample_init(init_key)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=init_key
    )

    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        model.logdensity_fn,
        L= L,
        step_size= step_size,
    )

    sampling_alg, transform = store_only_expectation_values(
                sampling_algorithm= sampling_alg,
                state_transform=frobenious(model), burnin = steps // 4)
    
    sampling_alg = thin_algorithm(sampling_alg, thinning = steps // num_saved_steps)

    initial_state = sampling_alg.init(initial_state)

    bias, info = run_inference_algorithm(
             rng_key=run_key,
             initial_state=initial_state,
             inference_algorithm=sampling_alg,
             num_steps= num_saved_steps,
             transform=transform,
             progress_bar=True,
         )[1]

    eevpd = jnp.std(info.energy_change)**2 / model.ndims
    
    return bias, eevpd



if __name__ == '__main__':

    model = StudentT(ndims= 100, dof= 5)
    bias, eevpd = run_mclmc(model, step_size= 2., L= 10, steps=10**4)

    jnp.savez('junk.npy', bias= bias, eevpd= eevpd)

#shifter --image=jrobnik/sampling:1.0 python3 -m papers.uHMC.identifying_failure