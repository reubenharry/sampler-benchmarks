
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
# from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_evaluation.models.phi4 import phi4
from sampler_comparison.samplers.annealing.annealing import annealed
import jax.numpy as jnp 
import numpy as np
from sampler_evaluation.models.u1 import U1


run_benchmarks(
        models={
            "U1": U1(Lt=4, Lx=4, beta=1),
        },
        samplers={

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=500),
            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            "nuts": partial(nuts,num_tuning_steps=500),
            # "nuts": partial(annealed(nuts, beta_schedule=[10.0, 5.0, 2.0],intermediate_num_steps=1000),num_tuning_steps=500),
            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=10000),
        },
        batch_size=batch_size,
        num_steps=2000,
        save_dir="results/U1",
        key=jax.random.key(20),
        map=jax.pmap
    )
