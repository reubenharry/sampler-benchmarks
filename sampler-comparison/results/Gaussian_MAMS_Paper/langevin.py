import os
import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
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


initial_position = jax.random.normal(jax.random.PRNGKey(0), (2,))

model = IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')
logdensity_fn = make_log_density_fn(model)

inverse_mass_matrix = jnp.eye(2)

initial_state = blackjax.langevin.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=jax.random.key(0),
        metric=metrics.default_metric(inverse_mass_matrix)
    )

run_benchmarks(
        models={
            # "Gaussian_MAMS_Paper": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
            "Gaussian_MAMS_Paper": model,
        },
        samplers={

            # "underdamped_langevin": lambda: unadjusted_lmc_no_tuning(
            #     initial_state=initial_state,
            #     integrator_type='velocity_verlet',
            #     step_size=0.1,
            #     L=1.,
            #     inverse_mass_matrix=inverse_mass_matrix,
            # ),
            "underdamped_langevin": lambda: unadjusted_lmc(desired_energy_var=1e-2, num_tuning_steps=5000, diagonal_preconditioning=False),
            # "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=500),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            # "nuts": lambda: nuts(num_tuning_steps=5000),
            # "unadj usted_microcanonical": lambda: unadjusted_mclmc(num_tuning_steps=20000),
        },
        batch_size=batch_size,
        num_steps=5000,
        save_dir="results/Gaussian_MAMS_Paper",
        key=jax.random.key(19),
        map=jax.pmap
    )
