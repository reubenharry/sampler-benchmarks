import os
import pprint
import jax
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from sampler_comparison.results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers import samplers
from sampler_evaluation.models import models
import pandas as pd
import jax.numpy as jnp


def test_samplers_expectations(key=jax.random.PRNGKey(1)):
    run_benchmarks(models=models, num_steps=50, batch_size=2, key=key)


def test_samplers_raw_samples(key=jax.random.PRNGKey(1)):

    for model, sampler in itertools.product(models, samplers):


        # model = "Banana"
        # sampler = "unadjusted_microcanonical"
        # sampler = "nuts"
        diagonal_preconditioning = False

        init_key, run_key = jax.random.split(key)

        initial_position = models[model].sample(seed=init_key)

        samplers[sampler](
            return_samples=True, diagonal_preconditioning=diagonal_preconditioning
        )(
            model=models[model],
            num_steps=50,
            initial_position=initial_position,
            key=run_key,
        )


if __name__ == "__main__":
    test_samplers_raw_samples()
    # test_samplers_expectations()
