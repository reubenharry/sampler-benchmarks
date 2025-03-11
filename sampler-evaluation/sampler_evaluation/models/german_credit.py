import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp

import os
module_dir = os.path.dirname(os.path.abspath(__file__))


def german_credit():

    german_credit = gym.targets.VectorModel(
        gym.targets.GermanCreditNumericSparseLogisticRegression(),
        flatten_sample_transformations=True,
    )

    with open(
        f"{module_dir}/data/{german_credit.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    
    german_credit.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: german_credit.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    german_credit.ndims = 51

    return german_credit
