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
from sampler_evaluation.models.gaussian import gaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc





run_benchmarks(
        models={
            "Gaussian": gaussian(ndims=100),
        },
        samplers={

            "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            # "nuts": lambda: nuts(num_tuning_steps=5000),
            # "unadjusted_microcanonical": lambda: unadjusted_mclmc(num_tuning_steps=20000),
        },
        batch_size=batch_size,
        num_steps=40000,
        save_dir="results/Gaussian",
        key=jax.random.key(19),
        map=jax.pmap
    )
