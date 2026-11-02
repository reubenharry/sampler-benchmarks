"""Trusted burn-in runner and fixed BlackJAX MCLMC production runner.

The runner owns the target evaluations, the counters and the phase schedule.
Submitted code receives only numerical observations.
"""
import jax
import jax.numpy as jnp

from blackjax.mcmc import mclmc

from evaluator.interface import (
    Observation, PhaseHistory, PhaseInfo, PublicConfig, Result, check_carry, identity_transform,
    phase_lengths,
)


def _observe(value_and_grad, carry, global_index, phase_index, key):
    position = carry["query_position"]
    logp, grad = value_and_grad(position)
    return Observation(
        position=jax.lax.stop_gradient(position),
        logp=jax.lax.stop_gradient(logp),
        grad=jax.lax.stop_gradient(grad),
        global_index=global_index,
        phase_index=phase_index,
        key=key,
    )


def _run_phase(submission, value_and_grad, phase, length, carry, offset, key):
    """Scan ``length`` active iterations of one phase; return carry and history."""
    keys = jax.random.split(key, length)

    def body(carry, inputs):
        index, step_key = inputs
        obs = _observe(value_and_grad, carry, offset + index, index, step_key)
        new_carry, _ = submission.step_phase(phase, carry, obs)
        return new_carry, PhaseHistory(obs.position, obs.logp, obs.grad)

    return jax.lax.scan(body, carry, (jnp.arange(length), keys))


def run_burn_in(submission, logdensity, x0, key, budget, hier_idx=(), latent_idx=()):
    """Run the phased burn-in of one chain.

    Parameters
    ----------
    submission : module
        Provides ``phase_fractions, init, step_phase, transition, finalize``.
    logdensity : callable
        Target log-density of a flat vector; never passed to the submission.
    x0 : array, shape (d,)
        Initial position.
    key : PRNGKey
        Chain key.
    budget : int
        Burn-in allowance ``B_i`` in gradient evaluations.
    hier_idx, latent_idx : int arrays
        Hierarchical and latent coordinate indices, passed to the submission in the public config.

    Returns
    -------
    Result, tuple of int
        Validated-by-caller result and the executed phase lengths.
    """
    config = PublicConfig(dim=int(x0.shape[0]), budget=int(budget),
                          hier_idx=tuple(int(i) for i in hier_idx), latent_idx=tuple(int(i) for i in latent_idx))
    lengths = phase_lengths(submission.phase_fractions(config), config.budget)
    assert sum(lengths) <= config.budget

    value_and_grad = jax.value_and_grad(logdensity)
    init_key, *phase_keys = jax.random.split(key, len(lengths) + 1)
    carry = submission.init(x0, init_key, config)
    check_carry(carry, config.dim)

    histories, offset = [], 0
    for phase, (length, phase_key) in enumerate(zip(lengths, phase_keys)):
        if length > 0:
            carry, history = _run_phase(
                submission, value_and_grad, phase, length, carry, offset, phase_key
            )
        else:
            empty = jnp.zeros((0, config.dim), x0.dtype)
            history = PhaseHistory(empty, jnp.zeros((0,), x0.dtype), empty)
        histories.append(history)
        offset += length
        if phase + 1 < len(lengths):
            info = PhaseInfo(phase=phase, length=length, num_phases=len(lengths))
            carry = submission.transition(carry, history, info)
            check_carry(carry, config.dim)

    result = submission.finalize(carry, tuple(histories), config)
    return Result(result.position, result.beta, result.transform_params), lengths


def run_production(result, logdensity, key, horizon, transform=identity_transform):
    """Run fixed BlackJAX MCLMC from ``result`` for at most ``horizon`` gradient evaluations.

    Cost: 1 evaluation to build the state plus 2 per transition (isokinetic
    McLachlan). Only completed transitions within the horizon are run.

    Returns
    -------
    positions : array, shape (n_steps, d)
        Positions ``z(w)`` in the original coordinates after each completed transition.
    costs : array, shape (n_steps,)
        Cumulative production gradient evaluations after each transition.
    """
    n_steps = max((int(horizon) - 1) // 2, 0)
    init_key, run_key = jax.random.split(key)
    logdensity_w = transformed_logdensity(logdensity, transform, result.transform_params)
    state = mclmc.init(result.position, logdensity_w, init_key)
    kernel = mclmc.build_kernel()
    beta = result.beta

    def body(state, step_key):
        state, _ = kernel(
            step_key, state, logdensity_w, beta.inverse_mass_matrix, beta.L, beta.step_size
        )
        return state, transform(result.transform_params, state.position)[0]

    _, positions = jax.lax.scan(body, state, jax.random.split(run_key, n_steps))
    costs = 1 + 2 * (jnp.arange(n_steps) + 1)
    return positions, costs


def transformed_logdensity(logdensity, transform, transform_params):
    """Fixed production target in the transformed coordinates, ``log p(z(w)) + logdet(w)``."""
    def logdensity_w(w):
        z, logdet = transform(transform_params, w)
        return logdensity(z) + logdet
    return logdensity_w


def run_production_errors(result, transform, logdensity, key, horizon, observable, ref_mean, ref_var):
    """Fixed BlackJAX MCLMC production returning the error curves after every transition.

    Sampling is done in the coordinates ``w`` of the submitted transform; the observable is
    evaluated on ``z(w)``. The running mean is updated online, so no samples are stored.
    Cost: 1 evaluation to build the state plus 2 per transition (isokinetic McLachlan);
    the transform itself is numerical and costs no target evaluation.

    Parameters
    ----------
    transform : callable
        ``transform(params, w) -> (z, logdet)``.
    observable : callable
        Maps a position ``z`` to the scored observable vector (squares of the
        constrained coordinates).
    ref_mean, ref_var : arrays
        Reference expectation and variance of the observable.

    Returns
    -------
    costs, error_mean, error_max : arrays of length n_steps
        Cumulative production evaluations and the coordinate-averaged and
        coordinate-maximum normalized squared error.
    """
    n_steps = max((int(horizon) - 1) // 2, 0)
    init_key, run_key = jax.random.split(key)
    logdensity_w = transformed_logdensity(logdensity, transform, result.transform_params)
    state = mclmc.init(result.position, logdensity_w, init_key)
    kernel = mclmc.build_kernel()
    beta = result.beta
    zero = jnp.zeros_like(observable(transform(result.transform_params, result.position)[0]))

    def body(carry, inputs):
        state, total = carry
        step_key, count = inputs
        state, _ = kernel(
            step_key, state, logdensity_w, beta.inverse_mass_matrix, beta.L, beta.step_size
        )
        total = total + observable(transform(result.transform_params, state.position)[0])
        errors = (total / count - ref_mean) ** 2 / ref_var
        return (state, total), (jnp.mean(errors), jnp.max(errors))

    counts = jnp.arange(1, n_steps + 1)
    _, (error_mean, error_max) = jax.lax.scan(
        body, (state, zero), (jax.random.split(run_key, n_steps), counts)
    )
    return 1 + 2 * counts, error_mean, error_max
