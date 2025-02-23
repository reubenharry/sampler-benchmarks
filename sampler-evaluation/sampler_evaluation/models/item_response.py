import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import pickle


def item_response():

    item_response = gym.targets.VectorModel(
        gym.targets.SyntheticItemResponseTheory(), flatten_sample_transformations=True
    )

    dirr = "/global/homes/r/reubenh/blackjax-benchmarks"
    with open(
        f"{dirr}/sampler-evaluation/sampler_evaluation/models/data/{item_response.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    item_response.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: item_response.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    item_response.ndims = 501

    return item_response
