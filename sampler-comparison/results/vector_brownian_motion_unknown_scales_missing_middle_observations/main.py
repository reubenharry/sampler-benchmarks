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
from sampler_evaluation.models.brownian import brownian_motion

model = brownian_motion()

# samplers_ulmc={

#             # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),
#             f"underdamped_langevin_{dev}": partial(unadjusted_lmc,desired_energy_var=dev, num_tuning_steps=20000, diagonal_preconditioning=True, stage3=False)

#             for dev in np.logspace(-6, -1, 15)
            
#             # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
# }

samplers={
            # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            # "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),

            # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            # "nuts": partial(nuts,num_tuning_steps=5000),

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000,),
            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=20000, num_windows=1),

            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
        }

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

