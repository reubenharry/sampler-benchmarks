import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import jax

import os

module_dir = os.path.dirname(os.path.abspath(__file__))

def banana():

    banana = gym.targets.Banana()

    try:
        with open(
            f"{module_dir}/data/{banana.name}_expectations.pkl",
            "rb",
        ) as f:
            stats = pickle.load(f)
    except:
        raise Exception(
            "Expectations not found: run estimate_expectations.py to generate them"
        )

    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    banana.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: banana.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )
    banana.ndims = 2

    def exact_sample(key):
        z = jax.random.normal(key, shape=(2,))
        x0 = 10.0 * z[0]
        x1 = 0.03 * (x0**2 - 100) + z[1]
        return jnp.array([x0, x1])

    banana.exact_sample = exact_sample

    return banana
