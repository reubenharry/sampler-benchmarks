"""The phased baseline adapter reproduces ``mclmc_find_L_and_step_size``."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import blackjax
from blackjax.mcmc import mclmc

from baseline.mclmc_adapter import make_baseline
from evaluator.runner import run_burn_in
from models.gaussian import diagonal_gaussian

jax.config.update("jax_enable_x64", True)


def _reference(logdensity, x0, key, num_steps):
    init_key, tune_key = jax.random.split(key)
    state = mclmc.init(x0, logdensity, init_key)
    kernel = mclmc.build_kernel()
    state, params, _ = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel, num_steps=num_steps, state=state, rng_key=tune_key,
        logdensity_fn=logdensity,
    )
    return state, params


@pytest.mark.parametrize("scales, num_steps, seed", [
    (np.geomspace(0.1, 10.0, 8), 400, 0),
    (np.geomspace(0.5, 2.0, 20), 600, 1),
    (np.ones(5), 300, 2),
])
def test_baseline_matches_blackjax(scales, num_steps, seed):
    model = diagonal_gaussian(scales)
    key = jax.random.key(seed)
    x0 = model.sample_init(jax.random.key(100 + seed))
    baseline = make_baseline(num_steps)

    result, lengths = run_burn_in(baseline, model.logdensity, x0, key, baseline.num_evaluations)
    assert sum(lengths) == baseline.num_evaluations

    submission_key = jax.random.split(key, 4)[0]  # the runner hands init the first of 1 + 3 split keys
    state, params = _reference(model.logdensity, x0, submission_key, num_steps)
    np.testing.assert_allclose(result.position, state.position, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.beta.step_size, params.step_size, rtol=1e-9)
    np.testing.assert_allclose(result.beta.L, params.L, rtol=1e-8)
    np.testing.assert_allclose(result.beta.inverse_mass_matrix, params.inverse_mass_matrix, rtol=1e-8)


def test_baseline_matches_blackjax_with_divergences():
    """Positive-orthant Gamma target: overshooting steps give NaN log-densities."""
    dim, num_steps = 6, 500

    def logdensity(x):
        return jnp.sum(jnp.log(x) - x / jnp.arange(1.0, dim + 1.0) * 0.5)

    x0 = jnp.linspace(0.5, 3.0, dim)
    key = jax.random.key(7)
    baseline = make_baseline(num_steps)
    result, _ = run_burn_in(baseline, logdensity, x0, key, baseline.num_evaluations)

    state, params = _reference(logdensity, x0, jax.random.split(key, 4)[0], num_steps)
    assert np.all(np.isfinite(state.position))
    np.testing.assert_allclose(result.position, state.position, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.beta.step_size, params.step_size, rtol=1e-8)
    np.testing.assert_allclose(result.beta.L, params.L, rtol=1e-6)
    np.testing.assert_allclose(result.beta.inverse_mass_matrix, params.inverse_mass_matrix, rtol=1e-6)
