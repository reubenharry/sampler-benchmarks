from typing import Callable
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax
import blackjax
from blackjax.util import pytree_size
from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)

from sampler_comparison.util import *


def da_adaptation(
    rng_key: PRNGKey,
    initial_position,
    algorithm,
    logdensity_fn: Callable,
    num_steps: int = 1000,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    integrator=blackjax.mcmc.integrators.velocity_verlet,
    # cos_angle_termination: float = 0.0,
):

    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    kernel = algorithm.build_kernel(integrator=integrator, 
                                    # cos_angle_termination=cos_angle_termination
                                    )
    init_kernel_state = algorithm.init(initial_position, logdensity_fn)
    inverse_mass_matrix = jnp.ones(pytree_size(initial_position))

    def step(state, key):


        adaptation_state, kernel_state = state
        # jax.debug.print("step {x}", x=jnp.exp(adaptation_state.log_step_size))

        # print("step size", jnp.exp(adaptation_state.log_step_size))

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
            info,
        )

    keys = jax.random.split(rng_key, num_steps)
    init_state = da_init(initial_step_size), init_kernel_state
    (adaptation_state, kernel_state), info = jax.lax.scan(
        step,
        init_state,
        keys,
    )
    return (
        kernel_state,
        {
            "step_size": da_final(adaptation_state),
            "inverse_mass_matrix": inverse_mass_matrix,
        },
        info,
    )
