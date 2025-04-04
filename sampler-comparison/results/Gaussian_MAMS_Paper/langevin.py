import os
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
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
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
import blackjax.mcmc.metrics as metrics
from functools import partial
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.banana import banana

model = banana()

initial_position = jax.random.normal(jax.random.PRNGKey(0), (model.ndims,))


# model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')
logdensity_fn = make_log_density_fn(model)


inverse_mass_matrix = jnp.eye(model.ndims)
initial_state = blackjax.langevin.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=jax.random.key(0),
        metric=metrics.default_metric(inverse_mass_matrix)
    )

run_benchmarks(
        models={
            # "Gaussian_MAMS_Paper": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
            "Banana": model,
            # "Brownian": model,
        },
        samplers={

            # "underdamped_langevin": partial(unadjusted_lmc_no_tuning,
            #     initial_state=initial_state,
            #     integrator_type='velocity_verlet',
            #     step_size=0.5,
            #     L=100.0,
            #     inverse_mass_matrix=inverse_mass_matrix,
            # ),

            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=5e-2, num_tuning_steps=5000, diagonal_preconditioning=False),
        },
        batch_size=batch_size,
        num_steps=20000,
        save_dir="results/Gaussian_MAMS_Paper",
        key=jax.random.key(19),
        map=jax.pmap
    )
