"""Oracle-by-oracle decomposition of one BlackJAX MCLMC transition.

One isokinetic McLachlan step needs two gradient evaluations. This module
advances a chain by one evaluation at a time, so every call of ``update``
consumes exactly one numerical observation ``(logp, grad)`` at the query point.
The arithmetic mirrors ``blackjax.mcmc.mclmc`` (isokinetic_mclachlan with the
Maruyama partial refreshment) and is tested against it.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackjax.mcmc.integrators import (
    esh_dynamics_momentum_update_one_step,
    mclachlan_coefficients,
    partially_refresh_momentum,
)
from blackjax.util import generate_unit_vector

B1, A1, B2 = mclachlan_coefficients[0], mclachlan_coefficients[1], mclachlan_coefficients[2]

STAGE_INIT = 0  # waiting for the observation at the initial point
STAGE_FIRST = 1  # waiting for the observation after the first drift
STAGE_SECOND = 2  # waiting for the observation after the second drift
STAGE_READY = 3  # transition closed; ``begin`` must start the next one


class Beta(NamedTuple):
    """MCLMC kernel parameters.

    Attributes
    ----------
    L : decoherence length.
    step_size : integrator step size.
    inverse_mass_matrix : diagonal inverse mass matrix, shape (d,).
    """

    L: jax.Array
    step_size: jax.Array
    inverse_mass_matrix: jax.Array


class StepperState(NamedTuple):
    """Chain state of the oracle-by-oracle MCLMC stepper."""

    position: jax.Array  # last accepted position
    momentum: jax.Array  # last accepted momentum
    logdensity: jax.Array
    logdensity_grad: jax.Array
    stage: jax.Array
    query: jax.Array  # point at which the next observation is requested
    flight_momentum: jax.Array  # momentum inside the running transition
    kinetic_change: jax.Array
    step_key: jax.Array  # key of the running transition
    beta: Beta  # parameters of the running transition


class StepInfo(NamedTuple):
    """Diagnostics of an oracle iteration; ``completed`` marks a finished transition."""

    completed: jax.Array
    energy_change: jax.Array
    nonans: jax.Array


def _momentum_update(momentum, grad, step_size, coef, inverse_mass_matrix, previous):
    update = esh_dynamics_momentum_update_one_step(inverse_mass_matrix)
    return update(momentum, grad, step_size, coef, previous, is_last_call=False)


def init_state(x0, beta, key):
    """Create a state whose first query is the initial position ``x0``."""
    dim = x0.shape[0]
    return StepperState(
        position=x0,
        momentum=generate_unit_vector(key, x0),
        logdensity=jnp.zeros((), x0.dtype),
        logdensity_grad=jnp.zeros(dim, x0.dtype),
        stage=jnp.asarray(STAGE_INIT),
        query=x0,
        flight_momentum=jnp.zeros(dim, x0.dtype),
        kinetic_change=jnp.zeros((), x0.dtype),
        step_key=key,
        beta=beta,
    )


def begin(state, beta, step_key):
    """Start a transition from the accepted state; returns the first drift query."""
    key_refresh_1, _ = jax.random.split(jax.random.split(step_key, 3)[0])
    momentum = partially_refresh_momentum(
        state.momentum, key_refresh_1, beta.step_size * 0.5, beta.L
    )
    momentum, kinetic_grad, kinetic_change = _momentum_update(
        momentum, state.logdensity_grad, beta.step_size, B1,
        beta.inverse_mass_matrix, None,
    )
    query = state.position + beta.step_size * A1 * kinetic_grad
    return state._replace(
        stage=jnp.asarray(STAGE_FIRST), query=query, flight_momentum=momentum,
        kinetic_change=kinetic_change, step_key=step_key, beta=beta,
    )


def _finish_step(state, logp, grad, randomize_on_failure=True):
    """Complete the transition after the second observation."""
    beta = state.beta
    kernel_key, _, nan_key = jax.random.split(state.step_key, 3)
    _, key_refresh_2 = jax.random.split(kernel_key)
    momentum, _, kinetic_change = _momentum_update(
        state.flight_momentum, grad, beta.step_size, B1,
        beta.inverse_mass_matrix, state.kinetic_change,
    )
    momentum = partially_refresh_momentum(
        momentum, key_refresh_2, beta.step_size * 0.5, beta.L
    )
    energy_change = kinetic_change - logp + state.logdensity
    nonans = (
        jnp.all(jnp.isfinite(state.query)) & jnp.all(jnp.isfinite(momentum))
        & jnp.isfinite(logp) & jnp.isfinite(kinetic_change)
        & jnp.isfinite(energy_change)
    )
    new_momentum = generate_unit_vector(nan_key, state.position)
    accepted = state._replace(
        position=state.query, momentum=momentum, logdensity=logp, logdensity_grad=grad,
    )
    rejected = state._replace(momentum=new_momentum if randomize_on_failure else state.momentum)
    new_state = jax.tree.map(lambda a, r: jnp.where(nonans, a, r), accepted, rejected)
    info = StepInfo(
        completed=jnp.asarray(True),
        energy_change=jnp.where(nonans, energy_change, 0.0),
        nonans=nonans,
    )
    return new_state, info


def advance(state, logp, grad, randomize_on_failure=True):
    """Absorb one observation without starting a new transition.

    After the second observation of a transition the returned stage is
    ``STAGE_READY`` and the caller must call ``begin`` in the same iteration.

    Parameters
    ----------
    state : StepperState
        State whose ``query`` produced ``(logp, grad)``.
    logp, grad : numerical observation at ``state.query``.
    randomize_on_failure : bool
        BlackJAX draws a fresh momentum after a failed transition; the tuning
        routine instead restores the previous momentum.

    Returns
    -------
    new_state, info
    """
    is_init = state.stage == STAGE_INIT
    is_first = state.stage == STAGE_FIRST
    ready = jnp.asarray(STAGE_READY)

    recorded = state._replace(logdensity=logp, logdensity_grad=grad, stage=ready)

    momentum, kinetic_grad, kinetic_change = _momentum_update(
        state.flight_momentum, grad, state.beta.step_size, B2,
        state.beta.inverse_mass_matrix, state.kinetic_change,
    )
    mid = state._replace(
        stage=jnp.asarray(STAGE_SECOND),
        query=state.query + state.beta.step_size * A1 * kinetic_grad,
        flight_momentum=momentum, kinetic_change=kinetic_change,
    )

    finished, finish_info = _finish_step(state, logp, grad, randomize_on_failure)
    finished = finished._replace(stage=ready)

    pick = lambda i, f, s: jnp.where(is_init, i, jnp.where(is_first, f, s))
    new_state = jax.tree.map(pick, recorded, mid, finished)
    waiting = is_init | is_first
    info = StepInfo(
        completed=~waiting,
        energy_change=jnp.where(waiting, 0.0, finish_info.energy_change),
        nonans=jnp.where(waiting, True, finish_info.nonans),
    )
    return new_state, info


def update(state, logp, grad, next_beta, next_key):
    """Advance one oracle iteration, starting the next transition when one closes.

    Parameters
    ----------
    state : StepperState
    logp, grad : numerical observation at ``state.query``.
    next_beta : Beta
        Parameters for a transition that may start in this iteration.
    next_key : PRNGKey
        Key of the transition that may start in this iteration.

    Returns
    -------
    new_state, info
    """
    advanced, info = advance(state, logp, grad)
    started = begin(advanced, next_beta, next_key)
    new_state = jax.tree.map(
        lambda b, a: jnp.where(advanced.stage == STAGE_READY, b, a), started, advanced
    )
    return new_state, info
