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
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc, adjusted_hmc_no_tuning
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)


model = IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')

initial_position = jax.random.normal(jax.random.PRNGKey(0), (model.ndims,))


logdensity_fn = make_log_density_fn(model)

inverse_mass_matrix = model.cov


initial_state = blackjax.dynamic_hmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=jax.random.key(0),
        # metric=metrics.default_metric(inverse_mass_matrix)
    )

print("begin")

samplers={

            # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-4, num_tuning_steps=20000, diagonal_preconditioning=True, stage3=False),
            "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=500, integrator_type="velocity_verlet", L_proposal_factor=1.25),
            "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=jnp.inf),
            # "adjusted_malt": partial(adjusted_hmc_no_tuning,integrator_type="velocity_verlet", L_proposal_factor=1.25, L=1, step_size=1e-1, inverse_mass_matrix=inverse_mass_matrix, initial_state=initial_state),
            
            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
}

run_benchmarks(

        models={"ICG_2D": model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=20000,
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