import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # defrags GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"      # don't grab all VRAM up front
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import jax.interpreters.xla as xla
import jax.core

if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
import os
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../blackjax')
sys.path.append('../sampler-comparison')
sys.path.append('../sampler-evaluation')
sys.path.append('../')
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sampler_evaluation
from sampler_comparison.samplers import samplers
import seaborn as sns
from functools import partial
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc


from sampler_comparison.samplers.general import make_log_density_fn
import blackjax
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

import numpy as np
import jax
import jax.numpy as jnp
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
import blackjax

from blackjax.adaptation.unadjusted_alba import unadjusted_alba
from blackjax.adaptation.unadjusted_step_size import robnik_step_size_tuning
from blackjax.adaptation.unadjusted_alba import unadjusted_alba
import math
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn
from functools import partial
from blackjax.adaptation.step_size import bisection_monotonic_fn
from blackjax.util import thin_algorithm
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc_no_tuning
from results.run_benchmarks import lookup_results


def compose(f, g):
    return lambda x: f(g(x))


def blackjax_las(model, num_chains, key, ndims, num_adjusted_steps, diagonal_preconditioning=False, target_acceptance_rate=0.8, target_eevpd=5e-1):

    init_key, tune_key, unadjusted_key, adjusted_key = jax.random.split(key, 4)
    initial_position = jax.random.normal(init_key, (ndims,))

    logdensity_fn = make_log_density_fn(model)
    ### Phase 1: unadjusted ###

        
    # burn-in and adaptation
    num_alba_steps = 10000
    warmup = unadjusted_alba(
        # algorithm=blackjax.mclmc, 
        mcmc_kernel=blackjax.mclmc.build_kernel(blackjax.mcmc.integrators.isokinetic_mclachlan),
        init=blackjax.mclmc.init,
        logdensity_fn=logdensity_fn, 
        target_eevpd=target_eevpd, 
        v=1, 
        num_alba_steps=num_alba_steps,
        preconditioning=diagonal_preconditioning,
        alba_factor=0.4,
        )

    (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, 20000)

    # sampling
    ess_per_sample = blackjax_mclmc_sampler_params['ESS']
    print(f"ESS per sample according to tuning: {ess_per_sample}")

    num_steps = math.ceil(num_chains // ess_per_sample)

    alg = blackjax.mclmc(
            logdensity_fn=logdensity_fn,
            L=blackjax_mclmc_sampler_params['L'],
            step_size=blackjax_mclmc_sampler_params['step_size'],
            inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )

    thinning_rate = math.ceil(1/ess_per_sample)

    jax.debug.print("thinning_rate {x}", x=thinning_rate)

    alg = thin_algorithm(
                    alg,
                    thinning=thinning_rate,
                    info_transform=lambda info: jax.tree.map(jnp.mean, info),
                    )

    


    final_output, (history) = run_inference_algorithm(
            rng_key=unadjusted_key,
            initial_state=blackjax_state_after_tuning,
            inference_algorithm=alg,
            num_steps=num_chains,
            # num_steps=num_chains*thinning_rate,
            transform=lambda a, b: a,
            progress_bar=False,
        )
    samples = history.position # [::thinning_rate]
    # jax.debug.print("shape of samples {x}", x=samples.shape)

    # adjusted_num_grads_per_step = 2 * num_adjusted_steps
    # subsamples = samples[::thinning_rate]
    unadjusted_num_grads_per_step = 2 * thinning_rate
    subsamples = samples
    # jax.debug.print("shape of subsamples {x}", x=subsamples.shape)
    # raise Exception("stop here")

    integration_steps_fn = make_random_trajectory_length_fn(True)



    def make_mams_step(key):
        def mams_step(inp):

            step_size, positions, info, step_size_adaptation_state = inp
            # jax.debug.print("step_size {step_size}", step_size=(step_size, blackjax_mclmc_sampler_params['L']))
            num_steps_per_traj = 15 # blackjax_mclmc_sampler_params['L'] / step_size
            
            keys = jax.random.split(key, positions.shape[0])
            alg = blackjax.adjusted_mclmc_dynamic(
                    logdensity_fn=logdensity_fn,
                    step_size=step_size,
                    integration_steps_fn=integration_steps_fn(num_steps_per_traj),
                    integrator=blackjax.mcmc.integrators.isokinetic_omelyan,
                    inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
                    # inverse_mass_matrix=(np.ones(positions.shape[1])),
                    L_proposal_factor=jnp.inf,
                )
            

            def step_fn(pos_key):
                pos, key = pos_key
                init_key, run_key = jax.random.split(key, 2)
                state, info = alg.step(
                    rng_key=run_key,
                    state=blackjax.adjusted_mclmc_dynamic.init(pos, logdensity_fn, init_key),
                )
                return state, info
            
            new_states, infos = jax.lax.map(step_fn, xs=(positions,keys))
            # jax.debug.print("num_integration_steps {x}", x=infos.num_integration_steps)
            return (step_size, new_states.position, infos, step_size_adaptation_state)
            # return (step_size, positions, infos, step_size_adaptation_state)

        return mams_step
        
    epsadap_update = bisection_monotonic_fn(target_acceptance_rate,tolerance=1e-4)
    step_size_adaptation_state_initial = (jnp.array([-jnp.inf, jnp.inf]), False)
        
    def tuning_step(inp):

        old_step_size, old_positions, old_infos, step_size_adaptation_state = inp
        acc_rate = old_infos.acceptance_rate.mean()
        # jax.debug.print("acc_rate {x}", x=old_infos.acceptance_rate)

        
        step_size_adaptation_state, new_step_size = epsadap_update(
            step_size_adaptation_state,
            old_step_size,
            acc_rate,
        )

        # new_step_size = 0.0
        
        return (new_step_size, old_positions, old_infos, step_size_adaptation_state)

    def step_fn(inp, key):
        results =  make_mams_step(key)(inp)
        tuned_params = tuning_step(results)
        return tuned_params, tuned_params
    
    initial_adjusted_key, adjusted_key = jax.random.split(adjusted_key, 2)

    _, _, infos, _ = make_mams_step(initial_adjusted_key)((0.1, subsamples, None, step_size_adaptation_state_initial))

    positions = subsamples
    step_size = blackjax_mclmc_sampler_params['step_size']

    _, (step_sizes, positions, infos, step_size_adaptation_state) = jax.lax.scan(step_fn, (step_size, subsamples, infos, step_size_adaptation_state_initial), jax.random.split(adjusted_key, num_adjusted_steps))

    # print(infos.num_integration_steps.shape, "num_integration_steps")
    adjusted_num_grads_per_step = infos.num_integration_steps.sum(axis=1)
    # print(adjusted_num_grads_per_step.shape, "adjusted_num_grads_per_step")

    return samples, positions, infos, num_steps, step_size_adaptation_state, step_sizes, unadjusted_num_grads_per_step, adjusted_num_grads_per_step


def las(num_adjusted_steps, num_chains, diagonal_preconditioning=False, target_acceptance_rate=0.8, target_eevpd=5e-4):

    def s(model, key):
        unadjusted_position, adjusted_position, infos, num_steps_unadjusted, step_size_adaptation_state, step_sizes, unadjusted_num_grads_per_step, adjusted_num_grads_per_step = blackjax_las(
            model=model,
            key=key,
            ndims=model.ndims,
            num_adjusted_steps=num_adjusted_steps,
            num_chains=num_chains,
            diagonal_preconditioning=diagonal_preconditioning,
            target_acceptance_rate=target_acceptance_rate,
            target_eevpd=target_eevpd
        )
        
        return unadjusted_position, adjusted_position, infos, num_steps_unadjusted, step_sizes, unadjusted_num_grads_per_step, adjusted_num_grads_per_step
        
    return s


def get_results(key, num_steps, batch_size, relative_path='./', target_eevpd=1e-3, dim=2, diagonal_preconditioning=False):

    model = IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log')

    mams_key, las_key = jax.random.split(key, 2)
    results = lookup_results(
            model=model, 
            key=jax.random.key(0),
            num_steps=num_steps, 
            mh=True, 
            canonical=False, 
            langevin=False, 
            tuning='alba', 
            integrator_type='omelyan', 
            diagonal_preconditioning=diagonal_preconditioning, 
            redo=False, 
            batch_size=batch_size, 
            relative_path=relative_path, 
            compute_missing=True,
            redo_bad_results=True,
            pseudofermion=False
        )
    
    mams_results = results[(results['max']==False) & (results['statistic']=='square')]['num_grads_to_low_error'].values[0]


    num_adjusted_steps = 50
    num_chains = 500

    sampler = las(num_adjusted_steps, num_chains, diagonal_preconditioning=diagonal_preconditioning, target_eevpd=target_eevpd, target_acceptance_rate=0.9)
    unadjusted_samples, adjusted_samples, infos, num_steps_unadjusted, step_sizes, unadjusted_num_grads_per_step, adjusted_num_grads_per_step = sampler(model, key=las_key)

    adjusted_error_at_each_step = ((((adjusted_samples**2).mean(axis=1) - model.sample_transformations["square"].ground_truth_mean[None, :])**2)/(model.sample_transformations["square"].ground_truth_standard_deviation[None, :]**2)).mean(axis=-1)


    adjusted_error_at_each_step = np.repeat(adjusted_error_at_each_step, np.ceil(infos.num_integration_steps.mean(axis=1)).astype(jnp.int32)*num_chains)

    cumulative_adjusted_error_at_each_step = np.cumsum(adjusted_error_at_each_step)/np.arange(1, adjusted_error_at_each_step.shape[0]+1)

    unadjusted_error_at_each_step = get_standardized_squared_error(
        unadjusted_samples[None, :, :], 
        f=model.sample_transformations["square"].fn,
        E_f=model.sample_transformations["square"].ground_truth_mean,
        Var_f=model.sample_transformations["square"].ground_truth_standard_deviation**2,
        contract_fn=jnp.mean
        )[0]

    unadjusted_error_at_each_step = np.repeat(unadjusted_error_at_each_step, unadjusted_num_grads_per_step)

    full = np.concatenate([unadjusted_error_at_each_step, adjusted_error_at_each_step])

    las_results = samples_to_low_error(full)

    with open(f'unadjusted_samples_{dim}_{eevpd}.pkl', 'wb') as f:
                pickle.dump(unadjusted_error_at_each_step, f)

    umclmc_results = samples_to_low_error(unadjusted_error_at_each_step)




    return mams_results, las_results, umclmc_results


if __name__ == "__main__":


   

        import pickle

        eevpds = [1e-3, 1e-2, 1e-1, 1e-0]
        dims = [100, 1000, 10000]
        full_results = {}
        import gc
        import itertools
        diagonal_preconditioning = False
        save_results = True
        for dim, eevpd in itertools.product(dims, eevpds):
            print(f"Running {eevpd} {dim}")

            mams_results, las_results, umclmc_results = get_results(jax.random.key(0), num_steps=1000, batch_size=10, target_eevpd=eevpd, dim=dim, diagonal_preconditioning=diagonal_preconditioning)
            full_results[(eevpd, dim)] = (mams_results, las_results, umclmc_results)
            # clear memory
            print(umclmc_results, "umclmc results")
            del mams_results, las_results, umclmc_results
            gc.collect()

            # save unadjusted_samples and adjusted_samples
            
        # save full_results
            if save_results:
                with open(f'full_results_new_{diagonal_preconditioning}.pkl', 'wb') as f:
                    pickle.dump(full_results, f)