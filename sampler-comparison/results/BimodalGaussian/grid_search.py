from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
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
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc, grid_search_unadjusted_lmc, grid_search_adjusted_mclmc

from sampler_evaluation.models.bimodal import bimodal_gaussian

model = bimodal_gaussian()


run_benchmarks(
        models={f"Bimodal": model},
        samplers={
            # "grid_search_malt": partial(grid_search_hmc, num_tuning_steps=500, integrator_type="velocity_verlet", num_chains=batch_size, L_proposal_factor=1.25),
            # "grid_search_hmc": partial(grid_search_hmc, num_tuning_steps=500, integrator_type="velocity_verlet", num_chains=batch_size),
            "grid_search_mams": partial(grid_search_adjusted_mclmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size),
            # "grid_search_unadjusted_lmc": partial(grid_search_unadjusted_lmc, num_tuning_steps=20000, integrator_type="velocity_verlet", num_chains=batch_size, opt='avg'),
            # "grid_search_adjusted_malt": partial(grid_search_hmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size, opt='avg'),
            },
        batch_size=batch_size,
        num_steps=20000,
        save_dir=f"results/BimodalGaussian",
        key=jax.random.key(20),
        map=lambda x : x,
        calculate_ess_corr=False,
    )
