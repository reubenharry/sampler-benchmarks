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
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc





run_benchmarks(
        models={
            "Stochastic_Volatility": stochastic_volatility(),
        },
        samplers={

            "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

            # "nuts": partial(nuts,num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=5e-4, num_tuning_steps=500, diagonal_preconditioning=False),
            # "underdamped_langevin": partial(unadjusted_lmc_no_tuning, step_size=1e-5, L=0.1, initial_state=initial_state,integrator_type='velocity_verlet', inverse_mass_matrix=jnp.ones((32,))),



            # "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=5000),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            # "nuts": lambda: nuts(num_tuning_steps=5000),
            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=25000, 
            #                                                       desired_energy_var=5e-6
            #                                                       ),
        },
        batch_size=batch_size,
        num_steps=500000,
        save_dir="results/Stochastic_Volatility",
        key=jax.random.key(19),
        map=jax.pmap
    )
