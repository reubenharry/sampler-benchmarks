import os
import pprint
import jax
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from results.run_benchmarks import run_benchmarks
from src.samplers import samplers
from src.models import models
from evaluation.ess import sampler_grads_to_low_error
import pandas as pd
import jax.numpy as jnp


def test_samplers_expectations(key=jax.random.PRNGKey(1)):
    run_benchmarks(num_steps=50, batch_size=2, key=key)


def test_samplers_raw_samples(key=jax.random.PRNGKey(1)):

    model = "Banana"
    # sampler = "unadjusted_microcanonical"
    sampler = "nuts"
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
