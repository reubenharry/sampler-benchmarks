from functools import partial
import itertools
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 512
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")


from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
import numpy as np
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc

dims = np.concatenate([np.arange(2,10), np.ceil(np.logspace(2,5, 5)).astype(int)])

# dims = [100, 1000, 10000]

integrator_types = ['velocity_verlet', 'mclachlan', 'omelyan']
# integrator_types = ['velocity_verlet']

for dim, integrator_type in itertools.product(dims, integrator_types):

    batch_size = min(4 + 1000 // dim, batch_size)

    print(f"Running for dim={dim}, integrator_type={integrator_type}, batch_size={batch_size}")

    run_benchmarks(
            models={
                f"Gaussian_{dim}": IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log'),
            },
            samplers={

                # f"grid_search_adjusted_microcanonical_{integrator_type}": lambda: grid_search_adjusted_mclmc(num_chains=batch_size, num_tuning_steps=5000, integrator_type=integrator_type),

                # f"grid_search_unadjusted_microcanonical_{integrator_type}": lambda: grid_search_unadjusted_mclmc(num_chains=batch_size, num_tuning_steps=10000, integrator_type=integrator_type),

                f"grid_search_hmc_{integrator_type}": partial(grid_search_hmc,num_chains=batch_size, num_tuning_steps=5000, opt='avg',integrator_type=integrator_type),
         
         
            },
            
            
            batch_size=batch_size,
            num_steps=10000,
            save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/grid_search",
            key=jax.random.key(19),
            map=lambda f:f
        )
    
    
