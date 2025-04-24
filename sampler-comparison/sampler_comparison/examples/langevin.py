import os
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
import jax
num_cores = jax.local_device_count()
import inference_gym.using_jax as gym


import jax.numpy as jnp
import matplotlib.pyplot as plt
# import sys
# sys.path.append('..')
# sys.path.append('./sampler-comparison')
# sys.path.append('../../')
# sys.path.append('../..')
from sampler_comparison.samplers import samplers
import seaborn as sns
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
import sampler_evaluation
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc_no_tuning
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc_no_tuning
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

# model=sampler_evaluation.models.Rosenbrock(D=1)
model = IllConditionedGaussian(ndims=2, condition_number=1000, eigenvalues='log')


initial_position = jax.random.normal(jax.random.PRNGKey(0), (model.ndims,))


logdensity_fn = make_log_density_fn(model)


# print(model.cov)

# inverse_mass_matrix = jnp.eye(model.ndims)
# inverse_mass_matrix = jnp.ones((model.ndims,))
inverse_mass_matrix = model.cov
# initial_state = blackjax.langevin.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         rng_key=jax.random.key(0),
#         metric=metrics.default_metric(inverse_mass_matrix)
#     )

# sampler = partial(unadjusted_lmc_no_tuning,
#                 initial_state=initial_state,
#                 integrator_type='velocity_verlet',
#                 step_size=5e-1,
#                 L=1e-1,
#                 inverse_mass_matrix=inverse_mass_matrix,
#             )

# initial_state = blackjax.dynamic_hmc.init(
#         position=initial_position,
#         logdensity_fn=logdensity_fn,
#         random_generator_arg=jax.random.key(0),
#         # metric=metrics.default_metric(inverse_mass_matrix)
#     )

initial_state = blackjax.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=jax.random.key(3),
        # metric=metrics.default_metric(inverse_mass_matrix)
    )
sampler = partial(unadjusted_mclmc_no_tuning,
                initial_state=initial_state,
                integrator_type='velocity_verlet',
                step_size=5e-1,
                L=5e-2,
                inverse_mass_matrix=inverse_mass_matrix,
            )

# sampler = samplers['nuts']

samples, metadata = sampler(return_samples=True)(
        model=model, 
        num_steps=1,
        initial_position=None, 
        key=jax.random.key(0))

print(samples[-1])