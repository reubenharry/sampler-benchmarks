import os
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
import jax
num_cores = jax.local_device_count()
import inference_gym.using_jax as gym


import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
sys.path.append('./sampler-comparison')
sys.path.append('../../blackjax')
from sampler_comparison.samplers import samplers
import seaborn as sns

import blackjax

from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model, make_log_density_fn
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc_no_tuning
import blackjax.mcmc.metrics as metrics
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc_no_tuning
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
import time

model=stochastic_volatility_mams_paper

# def adjusted_hmc_no_tuning(
#     initial_state,
#     integrator_type,
#     step_size,
#     L,
#     inverse_mass_matrix,
#     random_trajectory_length=True,
#     return_samples=False,
#     incremental_value_transform=None,
#     return_only_final=False,
# ):

ndims = model.ndims

logdensity_fn = make_log_density_fn(model)

initial_position = initialize_model(model, jax.random.PRNGKey(0))
initial_state = blackjax.dynamic_hmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=jax.random.key(0),
)
    



mclmc_initial_state = blackjax.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=jax.random.key(0),
)

toc = time.time()
samples, metadata = adjusted_mclmc_no_tuning(
    return_samples=True,
    L=2.,
    step_size=1.,
    integrator_type="velocity_verlet",
    inverse_mass_matrix=1.0,
    initial_state=mclmc_initial_state,
    )(
        model=model, 
        num_steps=10000,
        initial_position=jax.random.normal(jax.random.key(0), shape=(ndims,)), 
        key=jax.random.key(0))
tic = time.time()

print(tic-toc, "seconds")

toc = time.time()
samples, metadata = adjusted_hmc_no_tuning(
    return_samples=True,
    L=2.,
    step_size=1.,
    integrator_type="velocity_verlet",
    inverse_mass_matrix=jnp.eye(ndims),
    initial_state=initial_state,
    )(
        model=model, 
        num_steps=10000,
        initial_position=jax.random.normal(jax.random.key(0), shape=(ndims,)), 
        key=jax.random.key(0))
tic = time.time()

print(tic-toc, "seconds\n")