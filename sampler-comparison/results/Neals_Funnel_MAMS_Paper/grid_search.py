import os
import jax
batch_size = 32
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

from sampler_evaluation.models.neals_funnel_mams_paper import neals_funnel_mams_paper
from functools import partial
jax.config.update("jax_enable_x64", True)


import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
import blackjax.mcmc.metrics as metrics


from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.grid_search.grid_search import grid_search_adjusted_mclmc, grid_search_hmc
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc, grid_search_unadjusted_lmc
from sampler_evaluation.models.rosenbrock import Rosenbrock

model = Rosenbrock()


samplers={
            # "grid_search_hmc": partial(grid_search_hmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size),
            "grid_search_mams": partial(grid_search_adjusted_mclmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size, acc_rate=0.99),
            # "grid_search_unadjusted_lmc": partial(grid_search_unadjusted_lmc, num_tuning_steps=20000, integrator_type="velocity_verlet", num_chains=batch_size),
            }

run_benchmarks(
        models={model.name: model},
        samplers=samplers,
        batch_size=batch_size,
        num_steps=500000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=lambda x : x,
        calculate_ess_corr=False,
    )