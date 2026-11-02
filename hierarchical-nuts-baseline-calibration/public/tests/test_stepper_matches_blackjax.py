"""The oracle-by-oracle stepper reproduces the BlackJAX MCLMC kernel."""
import jax
import jax.numpy as jnp
import numpy as np

import blackjax
from blackjax.mcmc import mclmc

from evaluator.mclmc_stepper import Beta, init_state, update

jax.config.update("jax_enable_x64", True)


def _logdensity(x):
    scales = jnp.linspace(0.5, 3.0, x.shape[0])
    return -0.5 * jnp.sum((x / scales) ** 2) + 0.1 * jnp.sum(jnp.cos(x))


def test_matches_blackjax_kernel():
    dim, n_steps = 6, 40
    beta = Beta(jnp.asarray(2.3), jnp.asarray(0.7), jnp.linspace(0.3, 2.0, dim))
    x0 = jnp.linspace(-1.0, 2.0, dim)
    keys = jax.random.split(jax.random.key(3), n_steps + 2)

    kernel = mclmc.build_kernel()
    ref_state = mclmc.init(x0, _logdensity, keys[0])
    ref_positions, ref_energy = [], []
    for k in keys[1:n_steps + 1]:
        ref_state, info = kernel(k, ref_state, _logdensity, beta.inverse_mass_matrix, beta.L, beta.step_size)
        ref_positions.append(ref_state.position)
        ref_energy.append(info.energy_change)

    value_and_grad = jax.value_and_grad(_logdensity)
    state = init_state(x0, beta, keys[0])
    positions, energies, evals = [], [], 0
    begin_index = 1  # init iteration begins step 0 with keys[1]
    for _ in range(1 + 2 * n_steps):
        logp, grad = value_and_grad(state.query)
        evals += 1
        state, info = update(state, logp, grad, beta, keys[begin_index])
        if evals == 1 or bool(info.completed):
            begin_index += 1
        if bool(info.completed):
            positions.append(state.position)
            energies.append(info.energy_change)
    assert evals == 1 + 2 * n_steps
    np.testing.assert_allclose(np.array(positions), np.array(ref_positions), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.array(energies), np.array(ref_energy), rtol=1e-8, atol=1e-10)
