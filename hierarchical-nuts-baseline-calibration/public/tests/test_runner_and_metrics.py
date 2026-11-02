"""Runner accounting, schedule rules and scoring."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evaluator import metrics
from evaluator.interface import is_valid_result, phase_lengths
from evaluator.runner import run_burn_in, run_production
from examples import submission as minimal
from models.gaussian import diagonal_gaussian

jax.config.update("jax_enable_x64", True)


class CountingSubmission:
    """Two-phase submission that records the observation count in its carry."""

    def phase_fractions(self, config):
        return (0.25, 0.75)

    def init(self, x0, key, config):
        carry = minimal.init(x0, key, config)
        carry["count"] = jnp.asarray(0)
        carry["seen_global"] = jnp.asarray(-1)
        return carry

    def step_phase(self, phase, carry, obs):
        carry = dict(carry, count=carry["count"] + 1, seen_global=obs.global_index)
        return carry, {}

    def transition(self, carry, history, info):
        return carry

    def finalize(self, carry, history, config):
        sizes = [h.logp.shape[0] for h in history]
        position = jnp.array([carry["count"], carry["seen_global"], *sizes], dtype=jnp.float64)
        return minimal.finalize(dict(carry, production_position=position), history, config)


@pytest.mark.parametrize("budget", [0, 1, 7, 100])
def test_phase_lengths_never_exceed_budget(budget):
    lengths = phase_lengths((0.1, 0.0, 0.3, 0.6), budget)
    assert sum(lengths) <= budget and all(n >= 0 for n in lengths)
    assert sum(lengths) == budget  # fractions sum to 1


def test_invalid_fractions_rejected():
    for bad in [(0.7, 0.7), (-0.1, 0.5), (float("nan"),), ()]:
        with pytest.raises(ValueError):
            phase_lengths(bad, 10)


def _count_calls(budget, n_chains):
    calls = []
    model = diagonal_gaussian([1.0, 2.0, 3.0, 4.0])

    def target(x):
        jax.debug.callback(lambda: calls.append(1))
        return model.logdensity(x)

    sub = CountingSubmission()
    keys = jax.random.split(jax.random.key(0), n_chains)
    x0 = jnp.ones((n_chains, 4))
    run = jax.vmap(lambda x, k: run_burn_in(sub, target, x, k, budget)[0])
    out = run(x0, keys)
    jax.block_until_ready(out)
    jax.effects_barrier()
    return len(calls), out


def test_evaluation_count_across_phases_single_chain():
    n_calls, out = _count_calls(budget=40, n_chains=1)
    count, last_global, *sizes = np.asarray(out.position[0])
    assert count == 40 and last_global == 39  # phase boundary does not reset
    assert sizes == [10, 30]
    assert n_calls == 40


def test_zero_budget_makes_no_target_call():
    n_calls, out = _count_calls(budget=0, n_chains=3)
    assert n_calls == 0
    assert out.beta.step_size.shape == (3,)


def test_ten_chain_evaluation_count():
    n_calls, _ = _count_calls(budget=20, n_chains=10)
    print("callback invocations for 10 chains, B=20:", n_calls)
    assert n_calls in (20, 200)  # batched or per-chain callbacks; recorded in the report


def test_default_result_valid_and_production_cost():
    model = diagonal_gaussian([1.0, 2.0, 3.0])
    result, _ = run_burn_in(minimal, model.logdensity, jnp.ones(3), jax.random.key(1), 10)
    assert is_valid_result(result)
    positions, costs = run_production(result, model.logdensity, jax.random.key(2), 21)
    assert positions.shape == (10, 3) and int(costs[-1]) == 21
    positions, costs = run_production(result, model.logdensity, jax.random.key(2), 22)
    assert int(costs[-1]) == 21  # a transition that would exceed the horizon is not run


def test_first_crossing_is_brief_and_counts_prefix():
    errors = np.array([1.0, 0.005, 5.0, 0.5])
    costs = np.array([3, 5, 7, 9])
    assert metrics.first_crossing(errors, costs) == 5.0
    assert metrics.first_crossing(np.array([np.nan, 0.5]), costs[:2]) == np.inf
    assert metrics.first_crossing(np.array([0.01]), costs[:1]) == np.inf  # strictly below


@pytest.mark.parametrize("n_inf, expected_finite", [(4, True), (5, False), (10, False)])
def test_median_with_failed_chains(n_inf, expected_finite):
    counts = [100.0 + i for i in range(10 - n_inf)] + [np.inf] * n_inf
    assert np.isfinite(metrics.problem_count(counts)) == expected_finite


def test_scores():
    s = metrics.score([100, 200], [50, 400])
    assert np.isclose(s["F"], 1.0)
    s = metrics.score([100, 200], [50, np.inf])
    assert s["F"] == 0.0 and np.isclose(s["F_finite"], 2.0) and s["excluded_fraction"] == 0.5
    s = metrics.score([100], [np.inf])
    assert s["F"] == 0.0 and s["F_finite"] is None


def test_invalid_results_detected():
    from evaluator.interface import Result
    from evaluator.mclmc_stepper import Beta
    good = Beta(jnp.asarray(1.0), jnp.asarray(0.1), jnp.ones(3))
    assert is_valid_result(Result(jnp.zeros(3), good))
    for bad in [good._replace(step_size=jnp.asarray(0.0)), good._replace(L=jnp.asarray(jnp.inf)),
                good._replace(inverse_mass_matrix=jnp.array([1.0, -1.0, 1.0]))]:
        assert not is_valid_result(Result(jnp.zeros(3), bad))
    assert not is_valid_result(Result(jnp.array([0.0, jnp.nan, 0.0]), good))


def test_median_curve_crossing_differs_from_median_of_crossings():
    costs = np.arange(1, 6)
    errors = np.array([
        [0.5, 0.005, 0.5, 0.5, 0.5],   # brief crossing at 2
        [0.5, 0.5, 0.005, 0.005, 0.005],  # crossing at 3
        [0.5, 0.5, 0.5, 0.005, 0.005],  # crossing at 4
    ])
    per_chain = [metrics.first_crossing(e, costs) for e in errors]
    assert metrics.problem_count(per_chain) == 3.0
    assert metrics.median_curve_crossing(errors, costs) == 4.0  # median curve first below 0.01 at 4
    nan_curves = np.array([[np.nan] * 3, [0.5, 0.005, 0.005], [0.5, 0.005, 0.005]])
    assert metrics.median_curve_crossing(nan_curves, np.arange(1, 4)) == 2.0  # 2 of 3 chains suffice
    assert np.isinf(metrics.median_curve_crossing(np.array([[np.nan] * 3, [np.nan] * 3, [0.0] * 3]), np.arange(1, 4)))


def test_total_count_includes_burn_in_and_keeps_infinity():
    assert metrics.total_count(13335, 833) == 14168.0
    assert np.isinf(metrics.total_count(13335, np.inf))
