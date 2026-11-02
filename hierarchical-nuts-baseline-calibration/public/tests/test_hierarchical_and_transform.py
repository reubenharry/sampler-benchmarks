"""Hierarchical split, transforms in production, NUTS accounting and irregular-cost medians."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from baseline import nuts_baseline as nb
from evaluator import metrics
from evaluator.evaluate import Problem
from evaluator.interface import Result, identity_transform
from evaluator.mclmc_stepper import Beta
from evaluator.runner import run_production, run_production_errors
from models import hierarchical
from models.gaussian import diagonal_gaussian

jax.config.update("jax_enable_x64", True)


def _result(position, L=2.0, step=0.8, inv_mass=None, params=()):
    inv_mass = jnp.ones(position.shape[0]) if inv_mass is None else inv_mass
    return Result(position, Beta(jnp.asarray(L), jnp.asarray(step), inv_mass), params)


def test_funnel_split_partitions_coordinates():
    t = hierarchical.neal_funnel(12)
    assert set(t.hier_idx) | set(t.latent_idx) == set(range(12))
    assert not set(t.hier_idx) & set(t.latent_idx)


def test_funnel_with_data_reference_matches_mcmc():
    t = hierarchical.funnel_with_data(6, noise_std=1.0, seed=3)
    problem = Problem("f", 6, t.logdensity, lambda z: z, t.sample_init, t.ref_mean_x2, t.ref_var_x2, 1.0,
                      t.hier_idx, t.latent_idx)
    run = nb.run_nuts(problem, jax.random.key(0), 16, 500, 0.95, 400000)
    # NUTS second-moment estimates agree with the quadrature reference within Monte Carlo error
    final = run["error_max"][:, -1]
    assert np.median(final) < 0.05


def test_linear_transform_with_logdet_keeps_target():
    """Sampling w with z = a*w and logdet = d*log(a) reproduces second moments of z."""
    model = diagonal_gaussian([1.0, 2.0, 3.0])
    mean, var = model.reference()
    a = jnp.asarray([1.0, 2.0, 3.0])

    def transform(params, w):
        return params * w, jnp.sum(jnp.log(params))

    result = _result(jnp.zeros(3), params=a)
    positions, _ = run_production(result, model.logdensity, jax.random.key(0), 40001, transform)
    second = np.asarray(jnp.mean(positions ** 2, axis=0))
    assert np.all(np.abs(second - mean) < 0.2 * mean)


def test_wrong_logdet_changes_target():
    model = diagonal_gaussian([1.0, 1.0])

    def bad(params, w):
        return 2.0 * w, jnp.zeros(())  # missing log(2) is harmless, but a w-dependent one is not

    def bad_logdet(params, w):
        return w, -jnp.sum(w ** 2)

    result = _result(jnp.zeros(2))
    positions, _ = run_production(result, model.logdensity, jax.random.key(1), 40001, bad_logdet)
    assert float(jnp.mean(positions ** 2)) < 0.5  # tilted away from the true value 1


def _counted(model):
    calls = []

    def target(x):
        jax.debug.callback(lambda: calls.append(1))
        return model.logdensity(x)

    return target, calls


def test_production_cost_is_one_plus_two_per_transition_with_transform():
    target, calls = _counted(diagonal_gaussian([1.0, 2.0]))

    def transform(params, w):
        return 2.0 * w, jnp.asarray(2 * np.log(2.0))

    positions, costs = run_production(_result(jnp.zeros(2)), target, jax.random.key(0), 21, transform)
    jax.block_until_ready(positions)
    jax.effects_barrier()
    assert len(calls) == int(costs[-1]) == 21


def test_irregular_cost_median_matches_regular_case():
    errors = np.tile(np.array([1.0, 0.5, 0.2, 0.005, 0.004]), (5, 1))
    costs = np.array([3, 5, 7, 9, 11], dtype=float)
    regular = metrics.median_curve_crossing(errors, costs)
    irregular = metrics.median_curve_crossing(errors, np.tile(costs, (5, 1)))
    assert regular == 9.0
    assert abs(irregular - regular) < 0.01 * costs.max()


def test_irregular_cost_chain_offsets_and_failures():
    errors = np.array([[0.5, 0.001]] * 4 + [[np.inf, np.inf]] * 6)
    costs = np.array([[10.0, 20.0]] * 4 + [[10.0, 20.0]] * 6)
    assert metrics.median_curve_crossing(errors, costs) == np.inf  # 6 of 10 failed
    errors[4:] = [0.5, 0.001]  # now only 4 failed: median converges
    errors[:4] = np.inf
    assert np.isfinite(metrics.median_curve_crossing(errors, costs))


def test_nuts_warmup_cost_equals_executed_gradient_calls():
    target, calls = _counted(diagonal_gaussian([1.0, 2.0, 3.0]))
    state, params, cost = nb.warmup_chain(target, jnp.ones(3), jax.random.key(0), 50, 0.8)
    jax.block_until_ready(cost)
    jax.effects_barrier()
    assert len(calls) == int(cost)
