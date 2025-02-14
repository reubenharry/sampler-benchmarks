import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp


# def stochastic_volatility():
#     stochastic_volatility = gym.targets.stochastic_volatility()
#     stochastic_volatility.sample_transformations["square"] = model.Model.SampleTransformation(
#         fn=lambda params: params**2,
#         pretty_name="Square",
#         ground_truth_mean=jnp.array([100.0, 19.0]),
#         ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898])),
#     )

#     return stochastic_volatility


def stochastic_volatility():

    stochastic_volatility = gym.targets.VectorModel(
        gym.targets.VectorizedStochasticVolatilityLogSP500(),
        flatten_sample_transformations=True,
    )

    # with open(f"../sampler-evaluation/sampler_evaluation/models/data/{stochastic_volatility.name}_expectations.pkl", "rb") as f:
    #     stats = pickle.load(f)

    # e_x2 = stats['e_x2']
    # e_x4 = stats['e_x4']
    # var_x2 = e_x4 - e_x2**2

    # stochastic_volatility.sample_transformations["square"] = model.Model.SampleTransformation(
    #     fn=lambda params: params**2,
    #     pretty_name="Square",
    #     ground_truth_mean=e_x2,
    #     ground_truth_standard_deviation=jnp.sqrt(var_x2),
    # )

    stochastic_volatility.ndims = 2519

    return stochastic_volatility
