import pickle
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import numpy as np


def german_credit():

    german_credit = gym.targets.VectorModel(
        gym.targets.GermanCreditNumericSparseLogisticRegression(),
        flatten_sample_transformations=True,
    )

    dirr = "/global/homes/r/reubenh/blackjax-benchmarks"

    with open(
        f"{dirr}/sampler-evaluation/sampler_evaluation/models/data/{german_credit.name}_expectations.pkl",
        "rb",
    ) as f:
        stats = pickle.load(f)

    e_x = stats["e_x"]
    e_x2 = stats["e_x2"]
    e_x4 = stats["e_x4"]
    var_x2 = e_x4 - e_x2**2

    # import os
    # print("foo", os.listdir())

    # dirr = "../../MicroCanonicalHMC/benchmarks/"

    # e_x2, var_x2 = jnp.load(dirr + 'ground_truth/' + 'GermanCredit' + '/moments.npy')

    german_credit.sample_transformations["square"] = model.Model.SampleTransformation(
        fn=lambda params: german_credit.sample_transformations["identity"](params) ** 2,
        pretty_name="Square",
        ground_truth_mean=e_x2,
        ground_truth_standard_deviation=jnp.sqrt(var_x2),
    )

    # german_credit.sample_transformations["identity"] = (
    #     model.Model.SampleTransformation(
    #         fn=lambda params: params,
    #         pretty_name="Identity",
    #         ground_truth_mean=e_x,
    #         ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),
    #     )
    # )

    german_credit.ndims = 51

    return german_credit
