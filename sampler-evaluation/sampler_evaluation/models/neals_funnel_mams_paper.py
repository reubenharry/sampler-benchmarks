from collections import namedtuple
import os
import jax
jax.config.update("jax_enable_x64", True)
import sampler_evaluation

batch_size = 32
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys

sys.path.append(".")
sys.path.append("../sampler-evaluation")
import jax.numpy as jnp 

def logdensity_fn_funnel(x):
        # return 1.0

        """ x = [z_0, z_1, ... z_{d-1}, theta] """
        sigma_theta = 3.0
        ndims = 20
        theta = x[-1]
        X = x[..., :- 1]
        return -0.5* jnp.square(theta / sigma_theta) - 0.5 * (ndims - 1) * theta - 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1) 


def transform(x):
        """gaussianization"""
        xtilde = jnp.empty(x.shape)
        xtilde = xtilde.at[-1].set(x.T[-1] / 3.0)
        xtilde = xtilde.at[:-1].set(x.T[:-1] * jnp.exp(-0.5*x.T[-1]))
        return xtilde.T

# jax.config.update("jax_enable_x64", True)
SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation"])
Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations" ])
neals_funnel_mams_paper = Model(
    ndims = 20,
    log_density_fn=logdensity_fn_funnel,
    default_event_space_bijector=transform,
    sample_transformations={
        "identity": SampleTransformation(ground_truth_mean=jnp.ones(20)+jnp.inf, ground_truth_standard_deviation=3.0+jnp.inf),
        "square": SampleTransformation(ground_truth_mean=jnp.ones(20), ground_truth_standard_deviation=jnp.sqrt(2*jnp.ones(20))),
    },
)