from typing import Callable
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax

# from blackjax.adaptation.window_adaptation import da_adaptation

from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size, store_only_expectation_values
from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts_tuning import da_adaptation
from sampler_comparison.util import *


# produce a kernel that only stores the average values of the bias for E[x_2] and Var[x_2]
def with_only_statistics(
    model, alg, initial_state, rng_key, num_steps, incremental_value_transform=None
):

    if incremental_value_transform is None:
        incremental_value_transform = lambda x: jnp.array(
            [
                jnp.average(
                    jnp.square(
                        x[1] - model.sample_transformations["square"].ground_truth_mean
                    )
                    / (
                        model.sample_transformations[
                            "square"
                        ].ground_truth_standard_deviation
                        ** 2
                    )
                ),
                jnp.max(
                    jnp.square(
                        x[1] - model.sample_transformations["square"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "square"
                    ].ground_truth_standard_deviation
                    ** 2
                ),
                jnp.average(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / (
                        model.sample_transformations[
                            "identity"
                        ].ground_truth_standard_deviation
                        ** 2
                    )
                ),
                jnp.max(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "identity"
                    ].ground_truth_standard_deviation
                    ** 2
                ),
            ]
        )

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: jnp.array(
            [
                # model.sample_transformations["identity"](state.position),
                # model.sample_transformations["square"](state.position),
                state.position,
                state.position**2,
                state.position**4,
            ]
        ),
        incremental_value_transform=incremental_value_transform,
    )

    return run_inference_algorithm(
        rng_key=rng_key,
        initial_state=memory_efficient_sampling_alg.init(initial_state),
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )[1]
