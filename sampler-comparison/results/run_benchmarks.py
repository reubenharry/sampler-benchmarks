import os
import jax
import os
import jax
jax.config.update("jax_enable_x64", True)

from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from functools import partial
import pandas as pd
import os
import jax
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc_new, unadjusted_mclmc
from functools import partial
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.microcanonicalmontecarlo.mchmc import unadjusted_mchmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import unadjusted_hmc
import itertools
import jax.numpy as jnp
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc, grid_search_unadjusted_lmc, grid_search_hmc, grid_search_unadjusted_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.general import sampler_grads_to_low_error
from sampler_evaluation.models import models
import pandas as pd
import numpy as np

def run_benchmarks(
    models, samplers, batch_size, num_steps, key=jax.random.PRNGKey(1), save_dir=None,map=jax.pmap, calculate_ess_corr=False,
):

    for i, (sampler, model) in enumerate(itertools.product(samplers, models)):
        results = []

        key = jax.random.fold_in(key, i)

        (stats, _) = sampler_grads_to_low_error(
            sampler=map(
            lambda key, pos: samplers[sampler](return_samples=calculate_ess_corr)(
                model=models[model], 
                initial_position=pos, 
                key=key,
                num_steps=num_steps,
                
                )
            ),
            model=models[model],
            batch_size=batch_size,
            key=key,
            calculate_ess_corr=calculate_ess_corr
        )
        # jax.debug.print("stats {x}", x=1)


        for trans in models[model].sample_transformations:

            results.append(
                {
                    "Sampler": sampler,
                    "Model": model,
                    "num_grads_to_low_error": stats["max_over_parameters"][trans][
                        "grads_to_low_error"
                    ],
                    "grads_to_low_error_std": stats["max_over_parameters"][trans]["grads_to_low_error_std"],
                    "ess_corr": stats["max_over_parameters"][trans]["autocorrelation"],
                    "max": True,
                    "statistic": trans,
                    "num_tuning_grads": stats["num_tuning_grads"],
                    "L": stats["L"],
                    "step_size": stats["step_size"],
                    "batch_size": batch_size,
                }
            )
            results.append(
                {
                    "Sampler": sampler,
                    "Model": model,
                    "num_grads_to_low_error": stats["avg_over_parameters"][trans][
                        "grads_to_low_error"
                    ],
                    "grads_to_low_error_std": stats["avg_over_parameters"][trans]["grads_to_low_error_std"],
                    "ess_corr": stats["avg_over_parameters"][trans]["autocorrelation"],
                    "max": False,
                    "statistic": trans,
                    "num_tuning_grads": stats["num_tuning_grads"],
                    "L": stats["L"],
                    "step_size": stats["step_size"],
                    "batch_size": batch_size,
                }
            )

        # for trans in models[model].sample_transformations:
        #     jax.debug.print("transformation: {x}", x=trans)
        #     jax.debug.print("max (run benchmarks) {x}", x=stats["max_over_parameters"][trans]["grads_to_low_error"])
        #     jax.debug.print("avg (run benchmarks) {x}", x=stats["avg_over_parameters"][trans]["grads_to_low_error"])

        df = pd.DataFrame(results)

        if save_dir is not None:
            print(f"Saving results to", os.path.join(save_dir, f"{sampler}_{model}.csv"))
            df.to_csv(os.path.join(save_dir, f"{sampler}_{model}.csv"))


def lookup_results(model, batch_size, num_steps, mh : bool, canonical : bool, langevin : bool, tuning : str, integrator_type : str, diagonal_preconditioning : bool, redo : bool, relative_path : str = '.', compute_missing : bool = False, redo_bad_results : bool = None, statistic = 'square', key=jax.random.PRNGKey(16)):

    integrator_name = integrator_type.replace('_', ' ')


    unadjusted_tuning_steps = 20000
    adjusted_tuning_steps = 5000

    target_acc_rate = 0.9 if integrator_type == 'mclachlan' else 0.9

    sampler_dict = {

        # adjusted/unadjusted  canonical/microcanonical  langevin/nolangevin  alba/nuts
        (True, True, True, 'alba'): (f'adjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=adjusted_tuning_steps, integrator_type=integrator_type, L_proposal_factor=1.25,target_acc_rate=target_acc_rate, alba_factor=0.23, random_trajectory_length=False, diagonal_preconditioning=diagonal_preconditioning)),

        (True, True, False, 'alba'): (f'adjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=adjusted_tuning_steps, integrator_type=integrator_type, L_proposal_factor=jnp.inf,target_acc_rate=target_acc_rate,diagonal_preconditioning=diagonal_preconditioning)),

        (True, False, True, 'alba'): (f'adjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,L_proposal_factor=1.25, random_trajectory_length=False, alba_factor=0.23, target_acc_rate=target_acc_rate, num_tuning_steps=adjusted_tuning_steps,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (True, False, False, 'alba'): (f'adjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,num_tuning_steps=adjusted_tuning_steps,target_acc_rate=target_acc_rate,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, True, 'alba'): (f'unadjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=unadjusted_tuning_steps,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, False, 'alba'): (f'unadjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_hmc, desired_energy_var=3e-4,num_tuning_steps=unadjusted_tuning_steps,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, True, 'alba'): (f'unadjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mclmc,num_tuning_steps=unadjusted_tuning_steps, desired_energy_var=5e-4, diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, False, 'alba'): (f'unadjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mchmc,num_tuning_steps=unadjusted_tuning_steps, desired_energy_var=5e-4, diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),


        
        (True, True, False, 'nuts'): (f'adjusted_canonical_nolangevin_nuts_{integrator_name}_precond:{diagonal_preconditioning}', partial(nuts,num_tuning_steps=adjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, target_acc_rate=target_acc_rate)),
                                        # cos_angle_termination= cos_angle_termination)),


        
                    
    
        (False, False, True, 'grid_search'): (f'unadjusted_microcanonical_langevin_grid_search_{integrator_name}_precond:{diagonal_preconditioning}', partial(grid_search_unadjusted_mclmc_new,num_tuning_steps=unadjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, num_chains=batch_size)),
        
        (False, True, True, 'grid_search'): (f'unadjusted_canonical_langevin_grid_search_{integrator_name}_precond:{diagonal_preconditioning}', partial(grid_search_unadjusted_lmc,num_tuning_steps=unadjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, num_chains= batch_size)),
        
        (False, True, False, 'grid_search'): (f'unadjusted_canonical_nolangevin_grid_search_{integrator_name}_precond:{diagonal_preconditioning}', partial(grid_search_unadjusted_hmc,num_tuning_steps=unadjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, num_chains= batch_size)),
        
        (True, False, False, 'grid_search'): (f'adjusted_microcanonical_nolangevin_grid_search_{integrator_name}_precond:{diagonal_preconditioning}', partial(grid_search_adjusted_mclmc,num_tuning_steps=unadjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, num_chains= batch_size)),
                    
    
        (True, True, False, 'grid_search'): (f'adjusted_canonical_nolangevin_grid_search_{integrator_name}_precond:{diagonal_preconditioning}', partial(grid_search_hmc,num_tuning_steps=unadjusted_tuning_steps, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, num_chains= batch_size)),


    
        }
    
    results_dir = f'{relative_path}/results/{model.name}'

    # run_benchmarks(
    #     models={model.name: model},
    #     samplers={
    #         # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=20000, diagonal_preconditioning=True),
    #         "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
    #     },
    #     batch_size=batch_size,
    #     num_steps=20000,
    #     save_dir=results_dir,
    #     key=jax.random.key(20),
    #     map=jax.pmap,
    #     calculate_ess_corr=False,
    # )
    # raise Exception
    
    # load results
    # sampler_name, sampler = sampler_dict[(mh, canonical, langevin, tuning)]
    
    try:
        sampler_name, sampler = sampler_dict[(mh, canonical, langevin, tuning)]
    except KeyError:
        print(f"Sampler not found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning}")
        return pd.DataFrame()

    if redo:
        # remove the file
        if os.path.exists(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv')):
            os.remove(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))

    try:
        results = pd.read_csv(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))
        
        # Check if we need to rerun due to inf/nan in avg results
        if redo_bad_results is not None and not redo:  # Only check if redo_bad_results is True and we're not already redoing everything
            r_results = results[(results['max'] == (redo_bad_results == 'avg')) & (results['statistic'] == statistic)]  # Get only average results
            has_bad_values = r_results['num_grads_to_low_error'].apply(lambda x: pd.isna(x) or np.isinf(x) or np.isnan(x)).any()
            print(r_results['num_grads_to_low_error'], "has_bad_values")
            
            if has_bad_values:
                print(f"Found inf/nan in average results for {model.name}, rerunning...")
                if os.path.exists(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv')):
                    os.remove(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))
                raise FileNotFoundError  # This will trigger the rerun in the except block
        
        return results  # Return the loaded results
                
    except FileNotFoundError:
        print(f"File not found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning}, integrator_type={integrator_type}, diagonal_preconditioning={diagonal_preconditioning}")
        print(f'{sampler_name}_{model.name}.csv')

        if compute_missing:
            print(f"Creating file")
            
            # if results_dir does not exist, create it
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # run_benchmarks(
            #     models={model.name: model},
            #     samplers={sampler_name: sampler},
            #     batch_size=batch_size,
            #     num_steps=20000,
            #     save_dir=results_dir,
            #     key=jax.random.key(20),
            #     map=jax.pmap,
            #     calculate_ess_corr=False,
            # )

            # raise Exception
            if tuning == 'grid_search':
                map = lambda x : x
            else:
                map = jax.pmap

            # run sampler
            run_benchmarks(

                models={model.name: model},
                samplers={sampler_name: sampler},
                batch_size=batch_size,
                num_steps=num_steps,
                save_dir=results_dir,
                key=key,
                map=map,
                calculate_ess_corr=False,
            )
            print(f"Results saved to {results_dir}")

            results = pd.read_csv(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))
            return results  # Return the newly created results
        else:
            
            return pd.DataFrame()


if __name__ == "__main__":
    # model = gym.targets.dirichlet(dtype=jax.numpy.float64)
    # model = Dirichlet()
    # model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')

    run_benchmarks(models=models, samplers=samplers, batch_size=128, num_steps=10000)
