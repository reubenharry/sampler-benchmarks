import itertools
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
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
import numpy as np

dims = np.concatenate([np.arange(2,10), np.ceil(np.logspace(2,5, 5)).astype(int)])

integrator_types = ['velocity_verlet', 'mclachlan', 'omelyan']

for dim, integrator_type in itertools.product(dims, integrator_types):

    run_benchmarks(
            models={
                f"Gaussian_{dim}": IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log'),
            },
            samplers={

                f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=1000, integrator_type=integrator_type),
            
                f"unadjusted_microcanonical__{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=2000, integrator_type=integrator_type),

            },
            
    
            batch_size=batch_size,
            num_steps=2000,
            save_dir=f"sampler_comparison/experiments/dimensional_scaling",
            key=jax.random.key(19),
            map=jax.pmap
        )
