from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 512
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc

model = sampler_evaluation.models.Rosenbrock()

samplers={

            # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

            # "nuts": partial(nuts,num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-1, num_tuning_steps=20000),}

run_benchmarks(
        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=200000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=jax.pmap,
        calculate_ess_corr=False,
    )

# run_benchmarks(
#         models={
#             "Rosenbrock": sampler_evaluation.models.Rosenbrock(),
#         },
#         samplers={

#             "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

#             "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

#             "nuts": partial(nuts,num_tuning_steps=5000),

#             "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

#             "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=5e-1, num_tuning_steps=500, diagonal_preconditioning=False),

#             "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
#         },
#         batch_size=batch_size,
#         num_steps=40000,
#         save_dir="results/Rosenbrock",
#         key=jax.random.key(19),
#         map=jax.pmap
#     )
