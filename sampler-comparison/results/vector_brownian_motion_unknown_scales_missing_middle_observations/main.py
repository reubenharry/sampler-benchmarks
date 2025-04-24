from functools import partial
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
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
import blackjax.mcmc.metrics as metrics


from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers

model = sampler_evaluation.models.brownian_motion()

samplers={

            # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            # "nuts": partial(nuts,num_tuning_steps=5000),

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-1, num_tuning_steps=20000, diagonal_preconditioning=True, num_windows=1),

            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
        }

run_benchmarks(
        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=100000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=jax.pmap,
        calculate_ess_corr=False,
    )



# init_key = jax.random.key(0)

# model = brownian_motion()

# logdensity_fn = make_log_density_fn(model)
# initial_position = jax.random.normal(jax.random.PRNGKey(0), (model.ndims,))

# initial_state = blackjax.langevin.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         rng_key=init_key,
#         metric=blackjax.mcmc.metrics.default_metric(jnp.ones(initial_position.shape[0]))
#     )


# # model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')
# logdensity_fn = make_log_density_fn(model)


# inverse_mass_matrix = jnp.eye(model.ndims)
# initial_state = blackjax.langevin.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         rng_key=jax.random.key(0),
#         metric=metrics.default_metric(inverse_mass_matrix)
#     )

# run_benchmarks(
#         models={
#             "Brownian_Motion": model,

#         },
#         samplers={

#             "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

#             # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

#             # "nuts": partial(nuts,num_tuning_steps=5000),

#             # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

#             # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=5e-4, num_tuning_steps=500, diagonal_preconditioning=False),
#             # "underdamped_langevin": partial(unadjusted_lmc_no_tuning, step_size=1e-5, L=0.1, initial_state=initial_state,integrator_type='velocity_verlet', inverse_mass_matrix=jnp.ones((32,))),

#             # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
#         },
#         batch_size=batch_size,
#         num_steps=20000,
#         save_dir="results/Brownian_Motion",
#         key=jax.random.key(19),
#         map=jax.pmap
#     )
