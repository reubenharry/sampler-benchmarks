import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp


def brownian_motion():

    # load statistics from pickle
    # with open(f"./src/models/data/brownian_motion_expectations.pkl", "rb") as f:
    #     stats = pickle.load(f)

    # e_x2 = stats['e_x2']
    # e_x4 = stats['e_x4']
    # var_x2 = e_x4 - e_x2**2

    brownian_motion = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
    brownian_motion.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: params**2,
        pretty_name="Square",
        # ground_truth_mean=e_x2,
        # ground_truth_standard_deviation=jnp.sqrt(var_x2),
        ground_truth_mean=jnp.ones(32),
        ground_truth_standard_deviation=jnp.ones(32),
    )
    
    



    brownian_motion.ndims = 32

    return brownian_motion
