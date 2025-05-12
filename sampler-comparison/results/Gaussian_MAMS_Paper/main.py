import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
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
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
import numpy as np
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
import blackjax
import jax.numpy as jnp

model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')

# samplers_ulmc={

#             # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),
#             f"underdamped_langevin_{dev}": partial(unadjusted_lmc,desired_energy_var=dev, num_tuning_steps=20000, diagonal_preconditioning=True, stage3=False)

#             for dev in np.logspace(-6, -1, 15)
            
#             # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
# }

# samplers_mclmc = {
#             f"unadjusted_microcanonical_{dev}": partial(unadjusted_mclmc,num_tuning_steps=20000)
#             for dev in np.logspace(-6, -1, 15)
# }

# samplers = samplers_ulmc | samplers_mclmc


# init_key = jax.random.key(0)

# model = brownian_motion()

init_key = jax.random.key(0)
logdensity_fn = make_log_density_fn(model)
initial_position = jax.random.normal(jax.random.PRNGKey(0), (model.ndims,))

initial_state = blackjax.langevin.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
        metric=blackjax.mcmc.metrics.default_metric(jnp.ones(initial_position.shape[0]))
    )

inverse_mass_matrix = jnp.eye(model.ndims)
# initial_state = blackjax.langevin.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         rng_key=jax.random.key(0),
#         metric=metrics.default_metric(inverse_mass_matrix)
#     )

samplers = {
    # "underdamped_langevin": partial(unadjusted_lmc_no_tuning,desired_energy_var=1e-4,initial_state=initial_state, integrator_type="velocity_verlet", inverse_mass_matrix=inverse_mass_matrix, step_size=0.42, L=1.2),
    # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-4, num_tuning_steps=20000, diagonal_preconditioning=True),
    # "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),
    # "nuts": partial(nuts,num_tuning_steps=5000),
    # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
    "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=20000, diagonal_preconditioning=True),
    "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
}

run_benchmarks(

        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=100000,
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