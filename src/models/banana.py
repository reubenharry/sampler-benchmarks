import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp


def banana():

    # load statistics from pickle
    # with open(f"./src/models/data/banana_expectations.pkl", "rb") as f:
    #     stats = pickle.load(f)

    # e_x2 = stats['e_x2']
    # e_x4 = stats['e_x4']
    # var_x2 = e_x4 - e_x2**2

    banana = gym.targets.Banana()
    banana.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: params**2,
        pretty_name="Square",
        # ground_truth_mean=e_x2,
        # ground_truth_standard_deviation=jnp.sqrt(var_x2),
        ground_truth_mean=jnp.array([100.0, 19.0]),
        ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898])),
    )
    banana.ndims = 2

    return banana
