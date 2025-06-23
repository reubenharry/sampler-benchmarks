import os
import jax
<<<<<<< HEAD

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
=======
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
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
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from functools import partial
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.microcanonicalmontecarlo.mchmc import unadjusted_mchmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import unadjusted_hmc
import itertools
import jax.numpy as jnp

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

        df = pd.DataFrame(results)

        if save_dir is not None:
            print(f"Saving results to", save_dir)
            df.to_csv(save_dir)


<<<<<<< HEAD

# model = gym.targets.dirichlet(dtype=jax.numpy.float64)

# model = Dirichlet()


# model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')



def lookup_results(model, batch_size, 
                   mh : bool, 
                   canonical : bool, 
                   langevin : bool, 
                   tuning : str, 
                   integrator_type : str, 
                   diagonal_preconditioning : bool, 
                   cos_angle_termination: float = 0.,
                   ):
=======
def lookup_results(model, batch_size, num_steps, mh : bool, canonical : bool, langevin : bool, tuning : str, integrator_type : str, diagonal_preconditioning : bool, redo : bool, relative_path : str = '.'):
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
    
    integrator_name = integrator_type.replace('_', ' ')

    # note: num_tuning_steps is ill-named: for the adjusted samplers, it just controls how many tuning steps are used to find the mass matrix. 

    sampler_dict = {

        # adjusted/unadjusted  canonical/microcanonical  langevin/nolangevin  alba/nuts
        (True, True, True, 'alba'): (f'adjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=5000, integrator_type=integrator_type, L_proposal_factor=1.25,diagonal_preconditioning=diagonal_preconditioning)),

        (True, True, False, 'alba'): (f'adjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=5000, integrator_type=integrator_type, L_proposal_factor=jnp.inf,diagonal_preconditioning=diagonal_preconditioning)),

        (True, False, True, 'alba'): (f'adjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (True, False, False, 'alba'): (f'adjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,num_tuning_steps=5000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, True, 'alba'): (f'unadjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=20000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, False, 'alba'): (f'unadjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_hmc, desired_energy_var=3e-4,num_tuning_steps=20000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, True, 'alba'): (f'unadjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mclmc,num_tuning_steps=20000, desired_energy_var=5e-4, diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, False, 'alba'): (f'unadjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mchmc,num_tuning_steps=20000, desired_energy_var=5e-4, diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

<<<<<<< HEAD
        (False, False, True, 'alba'): (f'unadjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mclmc,num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, False, 'alba'): (f'unadjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mchmc,num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        
        
        (True, True, False, 'nuts'): (f'adjusted_canonical_nolangevin_nuts_{integrator_name}_precond:{diagonal_preconditioning}', partial(nuts,num_tuning_steps=500, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning, cos_angle_termination= cos_angle_termination)),
=======
        (True, True, False, 'nuts'): (f'adjusted_canonical_nolangevin_nuts_{integrator_name}_precond:{diagonal_preconditioning}', partial(nuts,num_tuning_steps=5000, integrator_type=integrator_type,diagonal_preconditioning=diagonal_preconditioning)),
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
                    }
    
    
    results_dir = f'{relative_path}/results/{model.name}'
    
    # load results
<<<<<<< HEAD
    sampler_name, sampler = sampler_dict[(mh, canonical, langevin, tuning)]
    
=======
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
    except FileNotFoundError:
        print(f"File not found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning}, integrator_type={integrator_type}, diagonal_preconditioning={diagonal_preconditioning}")

        print(f"Creating file")
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
        
    file = os.path.join(results_dir, f'{sampler_name}_{model.name}_cosangle={cos_angle_termination}.csv')

    if os.path.exists(file):
        os.remove(file)

<<<<<<< HEAD
    # if results_dir does not exist, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # run sampler
    run_benchmarks(

        models={model.name: model},
        samplers={sampler_name: sampler},
        batch_size=batch_size,
        num_steps=10000,
        save_dir=file,
        key=jax.random.key(19),
        map=jax.pmap,
        calculate_ess_corr=False,
    )
=======
            models={model.name: model},
            samplers={sampler_name: sampler},
            batch_size=batch_size,
            num_steps=num_steps,
            save_dir=results_dir,
            key=jax.random.key(19),
            map=jax.pmap,
            calculate_ess_corr=False,
        )
        print(f"Results saved to {results_dir}")
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b



if __name__ == "__main__":
    # model = gym.targets.dirichlet(dtype=jax.numpy.float64)
    # model = Dirichlet()
    # model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')

    run_benchmarks(models=models, samplers=samplers, batch_size=128, num_steps=10000)
