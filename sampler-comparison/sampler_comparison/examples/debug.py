from collections import namedtuple
import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

print(os.listdir())
import sys

# sys.path.append('..')
sys.path.append("../sampler-comparison")
# sys.path.append('../../')
from sampler_comparison.samplers import samplers

# import seaborn as sns

from sampler_evaluation.models import models
from sampler_evaluation.models.standardgaussian import Gaussian


import pymc as pm
import numpy as np

import pymc
from pymc.sampling.jax import get_jaxified_logp

# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer.reparam import TransformReparam

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

# Simulate outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

basic_model = pm.Model()

with basic_model as model:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0] * X1 + beta[1] * X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)


# def eight_schools_noncentered(J, sigma, y=None):
#     mu = numpyro.sample("mu", dist.Normal(0, 5))
#     tau = numpyro.sample("tau", dist.HalfCauchy(5))
#     with numpyro.plate("J", J):
#         with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
#             theta = numpyro.sample(
#                 "theta",
#                 dist.TransformedDistribution(
#                     dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
#                 ),
#             )
#         numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


# from numpyro.infer.util import initialize_model

# J = 8
# y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
# sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# rng_key, init_key = jax.random.split(jax.random.key(0))
# init_params, potential_fn_gen, *_ = initialize_model(
#     init_key,
#     eight_schools_noncentered,
#     model_args=(J, sigma, y),
#     dynamic_args=True,
# )

# logdensity_fn = lambda position: -potential_fn_gen(J, sigma, y)(position)
# init_position = init_params.z


Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])


def from_pymc(model):

    log_density_fn = get_jaxified_logp(model)

    rvs = [rv.name for rv in model.value_vars]

    init_position_dict = model.initial_point()
    init_position = [init_position_dict[rv] for rv in rvs]

    return (
        Model(
            ndims=len(init_position),
            log_density_fn=log_density_fn,
            default_event_space_bijector=lambda x: x,
        ),
        init_position,
    )


model, init_position = from_pymc(model)

samples, metadata = samplers["nuts"](return_samples=True)(
    model=model,
    num_steps=1000,
    initial_position=init_position,
    key=jax.random.PRNGKey(0),
)
