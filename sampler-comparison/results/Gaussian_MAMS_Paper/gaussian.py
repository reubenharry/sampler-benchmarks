import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 512
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
# from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from functools import partial
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_unadjusted_mclmc, grid_search_hmc




run_benchmarks(
        models={
            "Gaussian_MAMS_Paper": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
        },
        samplers={

            "hmc": partial(adjusted_hmc,num_tuning_steps=5000),
            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),
            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc, L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            # "nuts": partial(nuts, num_tuning_steps=5000),
            # "unadjusted_microcanonical": partial(unadjusted_mclmc, num_tuning_steps=20000),
        },
        batch_size=batch_size,
        num_steps=40000,
        save_dir="results/Gaussian_MAMS_Paper",
        key=jax.random.key(19),
        map=jax.pmap,
        calculate_ess_corr=False,
    )

# # dim = 100
# integrator_type = "velocity_verlet"
# run_benchmarks(
#             models={
#                 f"Gaussian_MAMS_Paper": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
#             },
#             samplers={

#                 # f"grid_search_adjusted_microcanonical_{integrator_type}": partial(grid_search_adjusted_mclmc,num_chains=batch_size, num_tuning_steps=5000, integrator_type=integrator_type, opt='avg'),

#                 f"grid_search_hmc_{integrator_type}": partial(grid_search_hmc,num_chains=batch_size, num_tuning_steps=5000, opt='avg',integrator_type=integrator_type),

   
         
#             },
            
            
#             batch_size=batch_size,
#             num_steps=10000,
#             save_dir=f"results/Gaussian_MAMS_Paper",
#             key=jax.random.key(19),
#             map=lambda f:f
#         )