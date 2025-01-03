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
from src.util import *

def da_adaptation(
    rng_key: PRNGKey,
    initial_position,
    algorithm,
    logdensity_fn: Callable,
    num_steps: int = 1000,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    integrator=blackjax.mcmc.integrators.velocity_verlet,
):

    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    kernel = algorithm.build_kernel(integrator=integrator)
    init_kernel_state = algorithm.init(initial_position, logdensity_fn)
    inverse_mass_matrix = jnp.ones(pytree_size(initial_position))

    def step(state, key):

        adaptation_state, kernel_state = state

        new_kernel_state, info = kernel(
            key,
            kernel_state,
            logdensity_fn,
            jnp.exp(adaptation_state.log_step_size),
            inverse_mass_matrix,
        )

        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )

        return (
            (new_adaptation_state, new_kernel_state),
            (True),
        )

    keys = jax.random.split(rng_key, num_steps)
    init_state = da_init(initial_step_size), init_kernel_state
    (adaptation_state, kernel_state), _ = jax.lax.scan(
        step,
        init_state,
        keys,
    )
    return kernel_state, {
        "step_size": da_final(adaptation_state),
        "inverse_mass_matrix": inverse_mass_matrix,
    }