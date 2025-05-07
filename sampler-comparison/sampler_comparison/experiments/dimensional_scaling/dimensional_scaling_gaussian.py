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
sys.path.append("../../blackjax-benchmarks/sampler-evaluation")


from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_unadjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
import numpy as np
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
import jax.numpy as jnp

dims = np.concatenate([np.arange(2,10), np.ceil(np.logspace(2,5, 5)).astype(int)])[12:]


# dims = [100000]

# integrator_types = ['velocity_verlet', 'mclachlan', 'omelyan']
integrator_types = ['velocity_verlet']

# for dim, integrator_type in itertools.product(dims, integrator_types):

#     batch_size = min(4 + 1000 // dim, batch_size)

#     print(f"Running for dim={dim}, integrator_type={integrator_type}, batch_size={batch_size}")


    
    # run_benchmarks(
    #         models={
    #             f"Gaussian_{dim}": IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log'),
    #         },
    #         samplers={

    #             # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=5000, integrator_type=integrator_type),

    #             # f"adjusted_hmc_{integrator_type}": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type=integrator_type, diagonal_preconditioning=True),

    #             f"underdamped_langevin_{integrator_type}": partial(unadjusted_lmc,desired_energy_var=1e-4, 
    #             # desired_energy_var_max_ratio=(1/desired_energy_var)*1000000,
    #             # desired_energy_var_max_ratio=jnp.inf,
                    
    #                 num_tuning_steps=20000, diagonal_preconditioning=False),

    #             # "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),

    #             # f"underdamped_langevin_{integrator_type}": partial(unadjusted_lmc,desired_energy_var=5e-2, num_tuning_steps=5000, diagonal_preconditioning=False, integrator_type=integrator_type),
            
    #             # f"unadjusted_microcanonical_{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=10000, integrator_type=integrator_type),

    #             # f"nuts_{integrator_type}": lambda: nuts(num_tuning_steps=5000),

    #         },
            
            
    #         batch_size=batch_size,
    #         num_steps=40000,
    #         save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/tuned/Gaussian",
    #         key=jax.random.key(19),
    #     )
for dim, integrator_type in itertools.product(dims, integrator_types):

    batch_size = min(4 + 1000 // dim, batch_size)

    print(f"Running for dim={dim}, integrator_type={integrator_type}, batch_size={batch_size}")


    
    run_benchmarks(
            models={
                f"Gaussian_{dim}": IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log', do_covariance=False),
            },
            samplers={

                # f"adjusted_microcanonical_{integrator_type}": lambda: adjusted_mclmc(num_tuning_steps=5000, integrator_type=integrator_type),

                # f"adjusted_hmc_{integrator_type}": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type=integrator_type, diagonal_preconditioning=True),

                "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=50, integrator_type="velocity_verlet", L_proposal_factor=1.25),

                # f"underdamped_langevin_{integrator_type}": partial(unadjusted_lmc,desired_energy_var=5e-2, num_tuning_steps=5000, diagonal_preconditioning=False, integrator_type=integrator_type),
            
                # f"unadjusted_microcanonical_{integrator_type}": lambda: unadjusted_mclmc(num_tuning_steps=10000, integrator_type=integrator_type),

                # f"nuts_{integrator_type}": lambda: nuts(num_tuning_steps=5000),

            },
            
            
            batch_size=batch_size,
            num_steps=5000,
            save_dir=f"sampler_comparison/experiments/dimensional_scaling/results/tuned/Gaussian",
            key=jax.random.key(19),
        )
    
