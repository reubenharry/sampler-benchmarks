# import inference_gym.using_jax as gym
import sys
# sys.path.append('')
sys.path.append('.')
sys.path.append('../sampler-comparison')
sys.path.append('../sampler-evaluation')
sys.path.append('../../')
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")

import os
# print(os.listdir('.'))
# raise Exception("Stop here")
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)


import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sampler_evaluation
from sampler_comparison.samplers import samplers
import seaborn as sns

from functools import partial
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc

# model= brownian_motion()

model = IllConditionedGaussian(1000,1)

sampler = partial(unadjusted_mclmc,integrator_type='velocity_verlet', diagonal_preconditioning=True, num_tuning_steps=10)

batch_size = 1

init_keys = jax.random.split(jax.random.key(3), batch_size)

keys = jax.random.split(jax.random.key(3), batch_size)

initial_position = jax.vmap(lambda key: initialize_model(model, key))(init_keys)

num_steps = 4000

samples, metadata = jax.pmap(
        lambda key, pos: sampler(return_samples=True)(
        model=model, num_steps=num_steps, initial_position=pos, key=key
        )
        )(
        keys,
        initial_position,
        )

print(metadata.keys())
print((metadata['info'].energy_change**2).mean())
