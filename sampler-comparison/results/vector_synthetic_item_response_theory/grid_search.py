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
from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper


model = sampler_evaluation.models.item_response()

run_benchmarks(
        models={model.name: model},
        samplers={"grid_search_hmc": partial(grid_search_hmc, num_tuning_steps=5000, integrator_type="velocity_verlet", num_chains=batch_size)},
        batch_size=batch_size,
        num_steps=100000,
        save_dir=f"results/{model.name}",
        key=jax.random.key(20),
        map=lambda x : x,
        calculate_ess_corr=False,
    )