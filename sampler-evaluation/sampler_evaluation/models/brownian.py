import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp

import os
module_dir = os.path.dirname(os.path.abspath(__file__))


def brownian_motion():

    brownian_motion = gym.targets.VectorModel(
        gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations(),
        flatten_sample_transformations=True,
    )


    with open(
        f"{module_dir}/data/{brownian_motion.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    e_x = stats["e_x"]
    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    brownian_motion.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: brownian_motion.sample_transformations["identity"](params)
        ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    brownian_motion.sample_transformations["identity"] = (
        model.Model.SampleTransformation(
            fn=lambda params: gym.targets.VectorModel(
                gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations(),
                flatten_sample_transformations=True, # TODO: sub out
            ).sample_transformations["identity"](params),
            pretty_name="Identity",
            ground_truth_mean=e_x,
            ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),
        )
    )

    brownian_motion.ndims = 32

    return brownian_motion
