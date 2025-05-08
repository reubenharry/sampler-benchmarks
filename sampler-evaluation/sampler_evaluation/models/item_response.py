import sys
sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import pickle

import os
module_dir = os.path.dirname(os.path.abspath(__file__))


def item_response():

    item_response = gym.targets.VectorModel(
        gym.targets.SyntheticItemResponseTheory(), flatten_sample_transformations=True
    )

    with open(
        f"{module_dir}/data/{item_response.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    # e_x2 = stats["e_x2"]
    # e_x4 = stats["e_x4"]

    e_x = stats["identity"]
    e_x2 = stats["square"]
    e_x4 = stats["quartic"]



    var_x2 = e_x4 - e_x2**2
    # e_x = stats["e_x"]
    cov = stats["covariance"]

    item_response.sample_transformations["identity"] = model.Model.SampleTransformation(
        fn=lambda params: params,
        pretty_name="Identity",
        ground_truth_mean=e_x,
        ground_truth_standard_deviation=jnp.sqrt(e_x2-e_x**2),
    )

    item_response.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: item_response.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )


    item_response.sample_transformations["quartic"] = model.Model.SampleTransformation(
        fn=lambda params: item_response.sample_transformations["identity"](params)
        ** 4,
        pretty_name="Quartic",
        ground_truth_mean=e_x4,
        ground_truth_standard_deviation=jnp.nan,
    )



    # item_response.sample_transformations["covariance"] = (
    #     model.Model.SampleTransformation(
    #         fn=lambda params: jnp.outer(item_response.sample_transformations["identity"](params) - e_x, item_response.sample_transformations["identity"](params) - e_x),
    #         pretty_name="Covariance",
    #         ground_truth_mean=cov,
    #         ground_truth_standard_deviation=jnp.nan,
    #     )
    # )

    item_response.ndims = 501

    return item_response
