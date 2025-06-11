from functools import partial
import itertools
import os

import jax
jax.config.update("jax_enable_x64", True)

batch_size = 2048
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")
sys.path.append("../../blackjax")
sys.path.append("../../blackjax-benchmarks/sampler-evaluation")

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
from sampler_comparison.samplers.grid_search.grid_search import grid_search_unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
import jax.numpy as jnp

Ds = np.concatenate([np.arange(2,10), np.ceil(np.logspace(2,4, 10)).astype(int)])[:]

def bias(expectation, f, model):
    return {
            'avg' : jnp.average((jnp.square(
                        expectation - model.sample_transformations[f].ground_truth_mean
                    ) / (
                        model.sample_transformations[f].ground_truth_mean
                        ** 2))[::2] ),
            'max' : jnp.max(jnp.square(
                        expectation - model.sample_transformations[f].ground_truth_mean
                    ) / (
                        model.sample_transformations[f].ground_truth_standard_deviation
                        ** 2)),
        }



# print(Ds)
# raise Exception


# Ds = [50, 500, 5000]

# print(Ds*2)
# raise Exception

# integrator_types = ['velocity_verlet', 'mclachlan', 'omelyan']
integrator_types = ['velocity_verlet'] # , 'mclachlan']

for D, integrator_type in itertools.product(Ds, integrator_types):

    dim = D*2
    batch_size = min(4 + 1000 // (dim), batch_size)

    print(f"Running for dim={dim}, integrator_type={integrator_type}, batch_size={batch_size}")

    model = Rosenbrock(D=dim)
    incremental_value_transform = lambda expectations: jax.tree.map_with_path(lambda path, expectation: 

    bias(expectation=expectation, f=path[0].key, model=model), expectations)


    run_benchmarks(
            models={
                f"Rosenbrock_{dim}": model,
            },
            samplers={

                f"unadjusted_microcanonical_{integrator_type}": partial(unadjusted_mclmc,num_tuning_steps=20000, integrator_type=integrator_type, incremental_value_transform=incremental_value_transform),




                f"underdamped_langevin_{integrator_type}": partial(unadjusted_lmc,desired_energy_var=1e-4, incremental_value_transform=incremental_value_transform),
                # desired_energy_var_max_ratio=(1/desired_energy_var)*1000000,
                # desired_energy_var_max_ratio=1e4,
                    
                    # num_tuning_steps=20000, diagonal_preconditioning=True, num_windows=1),
                f"adjusted_microcanonical_{integrator_type}": partial(adjusted_mclmc,num_tuning_steps=5000, integrator_type=integrator_type, incremental_value_transform=incremental_value_transform),

                # f"nuts_{integrator_type}": partial(nuts,num_tuning_steps=5000),

                f"adjusted_hmc_{integrator_type}": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type=integrator_type, incremental_value_transform=incremental_value_transform),

                "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25, incremental_value_transform=incremental_value_transform),
            

            },
            
            
            batch_size=batch_size,
            num_steps=50000,
            save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/tuned/Rosenbrock",
            key=jax.random.key(18),
            map=jax.pmap
        )
    
    # run_benchmarks(
    #         models={
    #             f"Rosenbrock_{dim}": Rosenbrock(D=D),
    #         },
    #         samplers={

    #             # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=1000, integrator_type=integrator_type),
            
    #             f"unadjusted_microcanonical__{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=50000, integrator_type=integrator_type),

    #         },
            
            
    #         batch_size=batch_size,
    #         num_steps=200000,
    #         save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/tuned/Rosenbrock",
    #         key=jax.random.key(19),
    #         map=jax.pmap
    #     )
    

    # run_benchmarks(
    #         models={
    #             f"Rosenbrock_{dim}": Rosenbrock(D=D),
    #         },
    #         samplers={

    #             f"grid_search_adjusted_microcanonical_{integrator_type}": lambda: grid_search_adjusted_mclmc(num_chains=batch_size, num_tuning_steps=500, integrator_type=integrator_type),

    #             # f"grid_search_unadjusted_microcanonical_{integrator_type}": lambda: grid_search_unadjusted_mclmc(num_chains=batch_size, num_tuning_steps=10000, integrator_type=integrator_type),


    #             # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=5000, integrator_type=integrator_type),
            
    #             # f"unadjusted_microcanonical__{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=20000, integrator_type=integrator_type),

    #         },
            
            
    #         batch_size=batch_size,
    #         num_steps=1000,
    #         save_dir=f"sampler_comparison/experiments/dimensional_scaling/results",
    #         key=jax.random.key(19),
    #         map=lambda f:f
    #     )
