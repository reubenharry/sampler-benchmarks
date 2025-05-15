from collections import namedtuple
import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp 

import os
from sampler_evaluation.models.model import SampleTransformation, make_model


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



neals_funnel_mams_paper = make_model(
    logdensity_fn=logdensity_fn_funnel,
    ndims=20,
    default_event_space_bijector=transform,
    sample_transformations={
        "identity": SampleTransformation(
                fn=lambda x: x,
                ground_truth_mean=jnp.ones(20)+jnp.inf, ground_truth_standard_deviation=3.0+jnp.inf),
        "square": SampleTransformation(
                fn=lambda x: x**2,
                ground_truth_mean=jnp.ones(20), ground_truth_standard_deviation=jnp.sqrt(2*jnp.ones(20))),
    },
    name='Neals_Funnel_MAMS_Paper',

)