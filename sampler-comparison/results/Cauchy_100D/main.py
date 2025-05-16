from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 64
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
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_evaluation.models.cauchy import cauchy
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc

samplers={

            "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=500, diagonal_preconditioning=False),
            "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000, diagonal_preconditioning=False),
            "nuts": partial(nuts,num_tuning_steps=500, diagonal_preconditioning=False),


            "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),
            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-4, num_tuning_steps=30000, diagonal_preconditioning=True),
            "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
        }

ndims = 100

model = cauchy(ndims=ndims)

run_benchmarks(
        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=200000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(19),
        map=jax.pmap
    )
