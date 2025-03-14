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
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_evaluation.models.phi4 import phi4
from sampler_comparison.samplers.annealing.annealing import annealed

L = 4
lam = 1

identity = lambda x:x

beta_schedule = [10.0, 5.0, 2.0]




run_benchmarks(
        models={
            f"Phi4_L{L}_lam{lam}": phi4(L=L, lam=lam),
        },
        samplers={

            "adjusted_microcanonical": lambda: annealed(adjusted_mclmc, beta_schedule=beta_schedule, num_tuning_steps=5000),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            # "nuts": lambda: nuts(num_tuning_steps=5000),
            # "unadjusted_microcanonical": lambda: unadjusted_mclmc(num_tuning_steps=10000),
        },
        batch_size=batch_size,
        num_steps=20000,
        save_dir=f"results/Phi4_L{L}_lam{lam}",
        key=jax.random.key(20),
        map=jax.pmap
    )
