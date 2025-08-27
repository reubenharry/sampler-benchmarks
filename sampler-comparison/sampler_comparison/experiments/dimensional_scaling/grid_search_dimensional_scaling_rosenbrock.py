from functools import partial
import itertools
import os

import jax
jax.config.update("jax_enable_x64", True)

batch_size = 1024
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
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
import numpy as np
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc, grid_search_unadjusted_lmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc


Ds = np.concatenate([np.arange(2,10), np.ceil(np.logspace(2,4, 5)).astype(int)])[9:]

# print(Ds)
# raise Exception

integrator_types = ['velocity_verlet'] # , 'mclachlan', 'omelyan']

for D, integrator_type in itertools.product(Ds, integrator_types):

    dim = D*2
    batch_size = min(4 + 1000 // (dim), batch_size)

    print(f"Running for dim={dim}, integrator_type={integrator_type}, batch_size={batch_size}")

    

    run_benchmarks(
            models={
                f"Rosenbrock_{dim}": Rosenbrock(D=D),
            },
            samplers={

                # f"grid_search_adjusted_microcanonical_{integrator_type}": lambda: grid_search_adjusted_mclmc(num_chains=batch_size, num_tuning_steps=5000, integrator_type=integrator_type),

                # f"grid_search_hmc_{integrator_type}": partial(grid_search_hmc,num_chains=batch_size, num_tuning_steps=5000, opt='avg',integrator_type=integrator_type),

                "grid_search_unadjusted_lmc": partial(grid_search_unadjusted_lmc, num_tuning_steps=20000, integrator_type="velocity_verlet", num_chains=batch_size, opt='avg')

                # f"grid_search_unadjusted_microcanonical_{integrator_type}": lambda: grid_search_unadjusted_mclmc(num_chains=batch_size, num_tuning_steps=10000, integrator_type=integrator_type),


                # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=5000, integrator_type=integrator_type),
            
                # f"unadjusted_microcanonical__{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=20000, integrator_type=integrator_type),

            },
            
            
            batch_size=batch_size,
            num_steps=40000,
            save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/grid_search/Rosenbrock",
            key=jax.random.key(19),
            map=lambda f:f
        )
    
    
    # run_benchmarks(
    #         models={
    #             f"Rosenbrock_{dim}": Rosenbrock(D=D),
    #         },
    #         samplers={

    #             # f"grid_search_adjusted_microcanonical_{integrator_type}": lambda: grid_search_adjusted_mclmc(num_chains=batch_size, num_tuning_steps=500, integrator_type=integrator_type),

    #             f"grid_search_unadjusted_microcanonical_{integrator_type}": lambda: grid_search_unadjusted_mclmc(num_chains=batch_size, num_tuning_steps=50000, integrator_type=integrator_type),


    #             # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=5000, integrator_type=integrator_type),
            
    #             # f"unadjusted_microcanonical__{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=20000, integrator_type=integrator_type),

    #         },
            
            
    #         batch_size=batch_size,
    #         num_steps=200000,
    #         save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/grid_search/Rosenbrock",
    #         key=jax.random.key(19),
    #         map=lambda f:f
    #     )
