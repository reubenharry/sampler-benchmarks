# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 512
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()


from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from functools import partial
# import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import pandas as pd
from sampler_evaluation.models.banana import banana
import os
from results.run_benchmarks import run_benchmarks
# import sampler_evaluation
from sampler_evaluation.models.dirichlet import Dirichlet
import jax
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from functools import partial
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.microcanonicalmontecarlo.mchmc import unadjusted_mchmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import unadjusted_hmc
import itertools
import jax.numpy as jnp

# model = gym.targets.dirichlet(dtype=jax.numpy.float64)

# model = Dirichlet()


# model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')



def lookup_results(model, mh : bool, canonical : bool, langevin : bool, tuning : str, integrator_type : str, diagonal_preconditioning : bool, redo : bool):
    
    integrator_name = integrator_type.replace('_', ' ')

    # make a dictionary that maps choice of parameters to a sampler. e.g. mh=True, canonical=True, tuning='none' -> 'nuts'
    sampler_dict = {
        # (True, True, True, 'nuts'): None,
        # (True, True, False, 'nuts'): (f'nuts_{integrator_type}', partial(nuts, integrator_type=integrator_type)),



        # adjusted/unadjusted  canonical/microcanonical  langevin/  alba
        (True, True, True, 'alba'): (f'adjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=500, integrator_type=integrator_type, L_proposal_factor=1.25,diagonal_preconditioning=diagonal_preconditioning)),

        (True, True, False, 'alba'): (f'adjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_hmc,num_tuning_steps=500, integrator_type=integrator_type, L_proposal_factor=jnp.inf,diagonal_preconditioning=diagonal_preconditioning)),

        (True, False, True, 'alba'): (f'adjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=500,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (True, False, False, 'alba'): (f'adjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(adjusted_mclmc,num_tuning_steps=500,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, True, 'alba'): (f'unadjusted_canonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, True, False, 'alba'): (f'unadjusted_canonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_hmc, desired_energy_var=3e-4,num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, True, 'alba'): (f'unadjusted_microcanonical_langevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mclmc,num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        (False, False, False, 'alba'): (f'unadjusted_microcanonical_nolangevin_alba_{integrator_name}_precond:{diagonal_preconditioning}', partial(unadjusted_mchmc,num_tuning_steps=2000,diagonal_preconditioning=diagonal_preconditioning, integrator_type=integrator_type)),

        
        
        # (True, True, False, 'tuning'): 'adjusted_hmc',
        # (True, False, False, 'alba') : ('adjusted_microcanonical', partial(adjusted_mclmc,num_tuning_steps=500))
        # (True, False, 'none'): 'adjusted_microcanonical_langevin',
        # (True, False, 'tuning'): 'adjusted_microcanonical_langevin',
        # (False, True, 'none'): 'adjusted_hmc',
        # (False, True, 'tuning'): 'adjusted_hmc',
                    }
    
    
    results_dir = f'results/{model.name}'
    
    # load results
    try:
        sampler_name, sampler = sampler_dict[(mh, canonical, langevin, tuning)]
    except KeyError:
        print(f"Sampler not found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning}")
        return pd.DataFrame()
        # raise ValueError(f"Sampler {sampler} not found")

    if redo:
        # remove the file
        if os.path.exists(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv')):
            os.remove(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))

    try:
        results = pd.read_csv(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))
        # display(results)
    except FileNotFoundError:
        print(f"File not found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning}, integrator_type={integrator_type}, diagonal_preconditioning={diagonal_preconditioning}")
        # ask user if they want to run the sampler
        # run_sampler = input(f"Run sampler {sampler_name} for {model.name}? (y/n)")
        # if run_sampler == 'y':

        # return pd.DataFrame()

        print(f"Creating file")
        
        # if results_dir does not exist, create it
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # run sampler
        run_benchmarks(

            models={model.name: model},
            samplers={sampler_name: sampler},
            batch_size=batch_size,
            num_steps=2000,
            save_dir=f"results/{model.name}",
            key=jax.random.key(19),
            map=jax.pmap,
            calculate_ess_corr=False,
        )
        print(f"Results saved to {results_dir}")

        results = pd.read_csv(os.path.join(results_dir, f'{sampler_name}_{model.name}.csv'))

    return results



mh_options = [True]
canonical_options = [True]
langevin_options = [False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet'] # , 'mclachlan', 'omelyan']
diagonal_preconditioning_options = [True]
models = [IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')]

redo = False 

full_results = pd.DataFrame()
for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    full_results = pd.concat([full_results, results], ignore_index=True)

