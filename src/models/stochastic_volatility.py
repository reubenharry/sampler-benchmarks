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
