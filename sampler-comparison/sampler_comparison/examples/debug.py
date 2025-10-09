import sys
sys.path.append('..')
sys.path.append('../sampler-comparison')
sys.path.append('../sampler-evaluation')
sys.path.append('../../')
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")

import jax
import jax.numpy as jnp
from sampler_evaluation.models.phi4 import phi4
from sampler_evaluation.models.data.estimate_expectations_phi4 import unreduce_lam
from sampler_comparison.samplers.general import with_only_statistics

import blackjax
from blackjax.util import run_inference_algorithm
# logdensity_fn = lambda x : -(jnp.sum(x**2))

from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc_no_tuning, unadjusted_mclmc
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc_no_tuning, adjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
side = 1024
model = phi4(side, unreduce_lam(reduced_lam=4.0, side=side), load_from_file=False)
logdensity_fn = model.log_density_fn
dim = model.ndims
# dim = 1000000
initial_position = jnp.ones(dim)
initial_state = blackjax.adjusted_mclmc_dynamic.init(initial_position, logdensity_fn, jax.random.key(0))

# alg = blackjax.mclmc(
#     logdensity_fn=logdensity_fn,
#     L=1.0,
#     step_size=0.1,
#     inverse_mass_matrix=jnp.ones(dim),
#     integrator=blackjax.mcmc.integrators.isokinetic_velocity_verlet,
# )

# alg_online, init, transform = with_only_statistics(
#                 model=model,
#                 alg=alg,
#                 incremental_value_transform=None,
#             )

# state = init(initial_state)

# final_output, history = run_inference_algorithm(
#             rng_key=jax.random.key(0),
#             initial_state=state,
#             inference_algorithm=alg_online,
#             num_steps=1000,
#             transform=(lambda a, b: (a,b)),
#             progress_bar=False,
#         )


# expectations, metadata = adjusted_mclmc_no_tuning(
#     initial_state=initial_state,
#     integrator_type="mclachlan",
#     step_size=0.1,
#     L=1.0,
#     inverse_mass_matrix=jnp.ones(dim),
# )(model, 50000, initial_position, jax.random.key(0))

import pickle
save = False
import time
tic = time.time()

if save:
    expectations, metadata = nuts(
        integrator_type="velocity_verlet",
        num_tuning_steps=5000,
        diagonal_preconditioning=False,
        # num_alba_steps=500,
        return_samples=True,
        return_only_final=True,
    )(model, 5000, initial_position, jax.random.key(0))


    # print(expectations.shape)

    # # pickle expectations
    import pickle
    with open('expectations.pkl', 'wb') as f:
        pickle.dump(expectations, f)
    # with open('metadata.pkl', 'wb') as f:
    #     pickle.dump(metadata, f)

else:

    with open('metadata.pkl', 'rb') as f:
        params = pickle.load(f)
        print(params.keys())
        step_size = params["step_size"]
        # inverse_mass_matrix = params["inverse_mass_matrix"]
        # integrator = params["L"]

    with open('expectations.pkl', 'rb') as f:
        initial_position = pickle.load(f)

    initial_state = blackjax.nuts.init(initial_position, logdensity_fn)

    toc = time.time()
    expectations, metadata = nuts(
        integrator_type="velocity_verlet",
        num_tuning_steps=0,
        diagonal_preconditioning=False,
        initial_state=initial_state,
        initial_step_size=step_size,
        initial_inverse_mass_matrix=jnp.ones(model.ndims),
    )(model, 200, initial_position, jax.random.key(0))

    

    jax.debug.print("expectations {x}", x=expectations['square']['avg'][0])
    jax.debug.print("metadata {x}", x=metadata)
    # alg = blackjax.nuts(
    #             logdensity_fn=logdensity_fn,
    #             step_size=step_size,
    #             inverse_mass_matrix=jnp.ones(model.ndims),
    #             integrator=blackjax.mcmc.integrators.velocity_verlet,
    #         )

    # rng_key, init_key = jax.random.split(jax.random.key(0))

    # alg, init, transform = with_only_statistics(
    #     model=model,
    #     alg=alg,
    #     incremental_value_transform=None,
    # )



    # state = init(initial_state)

    # get_final_sample = lambda output, info: (output[1][1], info)

    # final_output, history = run_inference_algorithm(
    #             rng_key=rng_key,
    #             initial_state=state,
    #             inference_algorithm=alg,
    #             num_steps=200,
    #             transform=transform,
    #             #progress_bar=progress_bar,
    #         )

    # jax.debug.print("final output {x}", x=final_output)
tic = time.time()
print(tic-toc, "seconds")

# expectations, metadata = (lambda key: adjusted_mclmc(
#     integrator_type="velocity_verlet",
#     num_tuning_steps=20000,
#     num_alba_steps=500,
# )(model, 5000, initial_position, key))(jax.random.split(jax.random.key(0)))

# state, info = alg.step(state=initial_state, rng_key=jax.random.key(0))

# print(metadata.keys())
# print(metadata['info'])

# print(final_output[1][1]['square'].shape)

# from collections import namedtuple
# import inference_gym.using_jax as gym
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import os

# print(os.listdir())
# import sys

# # sys.path.append('..')
# sys.path.append("../sampler-comparison")
# # sys.path.append('../../')
# from sampler_comparison.samplers import samplers

# # import seaborn as sns

# from sampler_evaluation.models import models
# from sampler_evaluation.models.standardgaussian import Gaussian


# import pymc as pm
# import numpy as np

# import pymc
# from pymc.sampling.jax import get_jaxified_logp

# # # import numpyro
# # # import numpyro.distributions as dist
# # # from numpyro.infer.reparam import TransformReparam

# # # True parameter values
# # alpha, sigma = 1, 1
# # beta = [1, 2.5]

# # # Size of dataset
# # size = 100

# # # Predictor variable
# # X1 = np.random.randn(size)
# # X2 = np.random.randn(size) * 0.2

# # RANDOM_SEED = 8927
# # rng = np.random.default_rng(RANDOM_SEED)

# # # Simulate outcome variable
# # Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

# # basic_model = pm.Model()

# # with basic_model as model:
# #     # Priors for unknown model parameters
# #     alpha = pm.Normal("alpha", mu=0, sigma=10)
# #     beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
# #     sigma = pm.HalfNormal("sigma", sigma=1)

# #     # Expected value of outcome
# #     mu = alpha + beta[0] * X1 + beta[1] * X2

# #     # Likelihood (sampling distribution) of observations
# #     Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)


# # # def eight_schools_noncentered(J, sigma, y=None):
# # #     mu = numpyro.sample("mu", dist.Normal(0, 5))
# # #     tau = numpyro.sample("tau", dist.HalfCauchy(5))
# # #     with numpyro.plate("J", J):
# # #         with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
# # #             theta = numpyro.sample(
# # #                 "theta",
# # #                 dist.TransformedDistribution(
# # #                     dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
# # #                 ),
# # #             )
# # #         numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


# # # from numpyro.infer.util import initialize_model

# # # J = 8
# # # y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
# # # sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# # # rng_key, init_key = jax.random.split(jax.random.key(0))
# # # init_params, potential_fn_gen, *_ = initialize_model(
# # #     init_key,
# # #     eight_schools_noncentered,
# # #     model_args=(J, sigma, y),
# # #     dynamic_args=True,
# # # )

# # # logdensity_fn = lambda position: -potential_fn_gen(J, sigma, y)(position)
# # # init_position = init_params.z


# # Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector"])


# # def from_pymc(model):

# #     log_density_fn = get_jaxified_logp(model)

# #     rvs = [rv.name for rv in model.value_vars]

# #     init_position_dict = model.initial_point()
# #     init_position = [init_position_dict[rv] for rv in rvs]

# #     return (
# #         Model(
# #             ndims=len(init_position),
# #             log_density_fn=log_density_fn,
# #             default_event_space_bijector=lambda x: x,
# #         ),
# #         init_position,
# #     )


# # model, init_position = from_pymc(model)

# # samples, metadata = samplers["nuts"](return_samples=True)(
# #     model=model,
# #     num_steps=1000,
# #     initial_position=init_position,
# #     key=jax.random.PRNGKey(0),
# # )
