import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import jax


def neals_funnel():

    ndims = 20
    neals_funnel = gym.targets.NealsFunnel(ndims=ndims)
    neals_funnel.ndims = ndims

    try:
        with open(
            f"../sampler-evaluation/sampler_evaluation/models/data/{neals_funnel.name}_expectations.pkl",
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

    neals_funnel.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: neals_funnel.sample_transformations["identity"](params)**2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    neals_funnel.exact_sample = lambda key: neals_funnel.sample(seed=key)

    return neals_funnel
