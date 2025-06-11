import pickle
import sys
#sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import jax

import os
module_dir = os.path.dirname(os.path.abspath(__file__))


def german_credit():

    gc = gym.targets.VectorModel(
        gym.targets.GermanCreditNumericSparseLogisticRegression(),
        flatten_sample_transformations=True,
    )

    with open(
        f"{module_dir}/data/{gc.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    # e_x2 = stats["e_x2"]
    # e_x4 = stats["e_x4"]
    # var_x2 = e_x4 - e_x2**2

    e_x = stats["identity"]
    e_x2 = stats["square"]
    e_x4 = stats["quartic"]
    cov = stats["covariance"]
    var_x2 = e_x4 - e_x2**2

    
    gc.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: gc.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    gc.sample_transformations["quartic"] = model.Model.SampleTransformation(
        fn=lambda params: gc.sample_transformations["identity"](params)
        ** 4,
        pretty_name="Quartic",
        ground_truth_mean=e_x4,
        ground_truth_standard_deviation=jnp.nan,
    )



    gc.sample_transformations["covariance"] = (
        model.Model.SampleTransformation(
            fn=lambda params: jnp.outer(gc.sample_transformations["identity"](params) - e_x, gc.sample_transformations["identity"](params) - e_x),
            pretty_name="Covariance",
            ground_truth_mean=cov,
            ground_truth_standard_deviation=jnp.nan,
        )
    )

    gc.ndims = 51


    def sample_init(key):
        weights = jax.random.normal(key, shape = (25, ))
        return jnp.concatenate((jnp.zeros(26), weights))
    
    gc.sample_init = sample_init

    return gc
