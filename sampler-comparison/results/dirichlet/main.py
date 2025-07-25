from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 2048
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers

from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts

import sys
sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
from sampler_evaluation.models.dirichlet import Dirichlet

# model = gym.targets.dirichlet(dtype=jax.numpy.float64)

model = Dirichlet()


samplers={

            # "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),
            # "adjusted_hmc_stage_2": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", stage_3=False),

            

            "nuts": partial(nuts,num_tuning_steps=5000),

            # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            # "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-1, num_tuning_steps=20000, diagonal_preconditioning=True),

            # "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
        }

run_benchmarks(
        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=40000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=jax.pmap,
        calculate_ess_corr=False,
    )
