"""NUTS baseline (BlackJAX 1.7.1) with chain-specific gradient-evaluation accounting.

Warmup is BlackJAX window adaptation (diagonal mass matrix, dual-averaging step size) at a
given target acceptance rate. One leapfrog step is one gradient evaluation, and building the
initial state costs one. The error curve of every chain is recorded against that chain's own
total cost (warmup plus production), because NUTS transitions cost a variable number of gradients.
"""
import jax
import jax.numpy as jnp
import numpy as np

import blackjax

from evaluator import metrics

ACCEPTANCE_RATES = (0.8, 0.9, 0.99)  # BlackJAX default, 0.9, 0.99
MAX_RECORDED_POINTS = 20000  # error curves are thinned to at most this many points per chain


def warmup_chain(logdensity, z0, key, num_warmup, target_acceptance_rate):
    """Window adaptation of one chain.

    Returns
    -------
    state, params, cost : final state, ``(step_size, inverse_mass_matrix)`` and warmup gradients.
    """
    adapt = blackjax.window_adaptation(
        blackjax.nuts, logdensity, is_mass_matrix_diagonal=True,
        target_acceptance_rate=target_acceptance_rate)
    (state, params), info = adapt.run(key, z0, num_steps=num_warmup)
    cost = 1 + jnp.sum(info.info.num_integration_steps)
    return state, params, cost


def production_chain(logdensity, observable, ref_mean, ref_var, state, params, key, num_blocks, stride):
    """Run ``num_blocks * stride`` NUTS transitions, recording error and cost every ``stride``.

    Returns
    -------
    state, error_max, error_mean, cost : state after the run and arrays of length ``num_blocks``
        (``cost`` is the cumulative number of production gradients at each record).
    """
    kernel = blackjax.nuts(logdensity, **params).step
    zero = jnp.zeros_like(ref_mean)

    def transition(carry, step_key):
        state, total, count, cost = carry
        state, info = kernel(step_key, state)
        total = total + observable(state.position)
        return (state, total, count + 1, cost + info.num_integration_steps), None

    def block(carry, block_key):
        carry, _ = jax.lax.scan(transition, carry, jax.random.split(block_key, stride))
        _, total, count, cost = carry
        errors = (total / count - ref_mean) ** 2 / ref_var
        return carry, (jnp.max(errors), jnp.mean(errors), cost)

    init = (state, zero, jnp.zeros((), ref_mean.dtype), jnp.zeros((), jnp.int64))
    (state, *_), (e_max, e_mean, cost) = jax.lax.scan(block, init, jax.random.split(key, num_blocks))
    return state, e_max, e_mean, cost


def warmup_costs(problem, key, num_chains, num_warmup, target_acceptance_rate):
    """Gradient cost of the warmup of ``num_chains`` chains (no production)."""
    init_key, warm_key = jax.random.split(key)
    z0 = jax.vmap(problem.sample_init)(jax.random.split(init_key, num_chains))
    warm = jax.jit(jax.vmap(lambda z, k: warmup_chain(
        problem.logdensity, z, k, num_warmup, target_acceptance_rate)))
    return np.asarray(warm(z0, jax.random.split(warm_key, num_chains))[2])


def run_nuts(problem, key, num_chains, num_warmup, target_acceptance_rate, horizon, stride=None):
    """Run ``num_chains`` independent NUTS chains until every chain has spent ``horizon`` total gradients.

    Parameters
    ----------
    problem : evaluator.evaluate.Problem
    horizon : int
        Total (warmup plus production) gradient evaluations to cover.
    stride : int, optional
        Transitions per recorded error value; default thins to ``MAX_RECORDED_POINTS``.

    Returns
    -------
    dict with ``error_max``, ``error_mean`` and ``total_cost`` (arrays ``(chains, n)``),
    ``warmup_cost`` (chains,), ``num_warmup``, ``target_acceptance_rate``.
    """
    init_key, warm_key, run_key = jax.random.split(key, 3)
    z0 = jax.vmap(problem.sample_init)(jax.random.split(init_key, num_chains))
    observable = lambda z: jnp.square(problem.constrain(z))
    ref_mean, ref_var = jnp.asarray(problem.ref_mean), jnp.asarray(problem.ref_var)

    warm = jax.jit(jax.vmap(lambda z, k: warmup_chain(
        problem.logdensity, z, k, num_warmup, target_acceptance_rate)))
    states, params, warmup_cost = warm(z0, jax.random.split(warm_key, num_chains))
    warmup_cost = np.asarray(warmup_cost)

    def chunk(states, keys, num_blocks, stride):
        run = jax.jit(jax.vmap(lambda s, p, k: production_chain(
            problem.logdensity, observable, ref_mean, ref_var, s, p, k, num_blocks, stride)))
        return run(states, params, keys)

    # Cost per transition is unknown in advance: run one block, estimate, then size the full run.
    probe_stride = 8
    probe_states, _, _, probe_cost = chunk(states, jax.random.split(run_key, num_chains), 1, probe_stride)
    per_transition = max(float(np.asarray(probe_cost).mean()) / probe_stride, 1.0)
    remaining = horizon - float(warmup_cost.min())
    transitions = max(int(np.ceil(1.5 * remaining / per_transition)), 1)
    stride = stride or max(1, transitions // MAX_RECORDED_POINTS)
    num_blocks = transitions // stride + 1
    _, e_max, e_mean, cost = chunk(states, jax.random.split(run_key, num_chains), num_blocks, stride)
    total_cost = warmup_cost[:, None] + np.asarray(cost)
    return {"error_max": np.asarray(e_max), "error_mean": np.asarray(e_mean), "total_cost": total_cost,
            "warmup_cost": warmup_cost, "num_warmup": num_warmup,
            "target_acceptance_rate": target_acceptance_rate, "stride": stride}


def crossing_count(run, horizon, threshold=metrics.THRESHOLD, key="error_max"):
    """Total gradient count at which the median-over-chains error first drops below ``threshold``.

    Only records within ``horizon`` total gradients are used. Returns ``inf`` if never.
    """
    cost = run["total_cost"]
    within = cost.min(axis=0) <= horizon
    return metrics.median_curve_crossing(run[key][:, within], cost[:, within], threshold)
