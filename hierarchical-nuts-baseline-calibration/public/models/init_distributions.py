"""Initialization distributions of the benchmark problems.

Unless stated otherwise the initial point is drawn from N(0, std^2) in the
unconstrained coordinates, with the width differing per problem (narrow, unit,
wide) and a log-spaced anisotropic width for two problems.
"""
import jax
import jax.numpy as jnp
import numpy as np

# name -> isotropic std, or ("anisotropic", low, high, seed) for log-spaced widths
INIT = {
    "ill_conditioned_gaussian": 3.0,
    "banana": 3.0,
    "quartic": 3.0,
    "eight_schools": 1.0,
    "brownian_unknown_scales": 2.0,
    "brownian_missing_middle": 0.3,
    "stochastic_volatility_small": 1.0,
    "stochastic_volatility": 0.3,
    "log_gaussian_cox": 2.0,
    "item_response": 0.5,
    "german_credit_logistic": 1.0,
    "german_credit_probit": 0.3,
    "german_credit_sparse": ("anisotropic", 0.1, 3.0, 11),
    "radon_indiana": 1.0,
    "radon_minnesota": ("anisotropic", 0.2, 2.0, 12),
    "radon_halfnormal_indiana": 0.5,
    "phi4_L8_lam5.198": 0.5,
    "phi4_L16_lam4.344": 2.0,
}


def init_std(name, dim):
    """Per-coordinate standard deviation of the initialization distribution."""
    spec = INIT.get(name)
    if spec is None:
        return None
    if isinstance(spec, tuple):
        _, low, high, seed = spec
        stds = np.geomspace(low, high, dim)
        return np.random.RandomState(seed).permutation(stds)
    return np.full(dim, float(spec))


def with_init(target):
    """Return ``target`` with its catalog initialization distribution (unchanged if none is listed)."""
    std = init_std(target.name, target.dim)
    if std is None:
        return target
    std = jnp.asarray(std)
    return target._replace(sample_init=lambda key: std * jax.random.normal(key, (target.dim,), jnp.float64))
