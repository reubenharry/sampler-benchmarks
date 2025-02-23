import os
import pprint
import jax
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from sampler_comparison.results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.general import initialize_model
import pandas as pd
import jax.numpy as jnp

from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.standardgaussian import Gaussian

models = {
    "Banana": banana(),
    "Gaussian_100D": Gaussian(ndims=100),
    "Brownian_Motion": brownian_motion(),
}


def test_samplers_expectations(key=jax.random.PRNGKey(1)):
    run_benchmarks(
        models=models, samplers=samplers, num_steps=50, batch_size=2, key=key
    )


def test_samplers_raw_samples(key=jax.random.PRNGKey(1)):

    for model, sampler in itertools.product(models, samplers):

        diagonal_preconditioning = False

        init_key, run_key = jax.random.split(key)

        initial_position = initialize_model(models[model], init_key)

        samples, _ = samplers[sampler](
            return_samples=True,
            diagonal_preconditioning=diagonal_preconditioning,
            num_tuning_steps=200,
        )(
            model=models[model],
            num_steps=50,
            initial_position=initial_position,
            key=run_key,
        )
        assert jnp.sum(jnp.any(jnp.isnan(samples))) == 0

        del samples


if __name__ == "__main__":
    test_samplers_raw_samples()
    test_samplers_expectations()
