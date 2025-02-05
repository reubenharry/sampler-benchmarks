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
from sampler_evaluation.evaluation.ess import calculate_ess


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


def sampler_grads_to_low_error(
    sampler, model, num_steps, batch_size, key, pvmap=jax.vmap
):

    try:
        model.sample_transformations[
            "square"
        ].ground_truth_mean, model.sample_transformations[
            "square"
        ].ground_truth_standard_deviation
    except:
        raise AttributeError("Model must have E_x2 and Var_x2 attributes")

    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch_size)

    squared_errors, metadata = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=num_steps, initial_position=pos, key=key
        )
    )(
        jax.random.normal(
            shape=(
                batch_size,
                model.ndims,
            ),
            key=init_key,
        ),
        keys,
    )
    # TODO: propoer initialization!

    err_t_avg_x2 = jnp.median(squared_errors[:, :, 0], axis=0)
    _, grads_to_low_avg_x2, _ = calculate_ess(
        err_t_avg_x2,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_max_x2 = jnp.median(squared_errors[:, :, 1], axis=0)
    _, grads_to_low_max_x2, _ = calculate_ess(
        err_t_max_x2,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_avg_x = jnp.median(squared_errors[:, :, 2], axis=0)
    _, grads_to_low_avg_x, _ = calculate_ess(
        err_t_avg_x,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_max_x = jnp.median(squared_errors[:, :, 3], axis=0)
    _, grads_to_low_max_x, _ = calculate_ess(
        err_t_max_x,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    return (
        {
            "max_over_parameters": {
                "square": {
                    "error": err_t_max_x2,
                    "grads_to_low_error": grads_to_low_max_x2.item(),
                },
                "identity": {
                    "error": err_t_max_x,
                    "grads_to_low_error": grads_to_low_max_x.item(),
                },
            },
            "avg_over_parameters": {
                "square": {
                    "error": err_t_avg_x2,
                    "grads_to_low_error": grads_to_low_avg_x2.item(),
                },
                "identity": {
                    "error": err_t_avg_x,
                    "grads_to_low_error": grads_to_low_avg_x.item(),
                },
            },
            "num_tuning_grads": metadata["num_tuning_grads"].mean().item(),
            "L": metadata["L"].mean().item(),
            "step_size": metadata["step_size"].mean().item(),
        },
        squared_errors,
    )