# a class that inherits from the inference_gym Model class but adds expectations of x^2

import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp


def ill_conditioned_gaussian(
    ndims=100,
    gamma_shape_parameter=0.5,
    max_eigvalue=None,
):
    icg = gym.targets.ill_conditioned_gaussian()
    icg.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: params**2,
        pretty_name="Square",
        ground_truth_mean=todo,
        ground_truth_standard_deviation=todo,
    )
    return icg


# print(banana.sample_transformations)

# print(banana.default_event_space_bijector(jnp.array([100.0, 19.0])))
