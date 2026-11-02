"""Hierarchical structure of the benchmark targets.

Every problem splits its unconstrained coordinates into *hierarchical* parameters
(few: hyper-, scale- and other global parameters) and *latent* parameters (many:
group-, site- or time-specific variables). The split is given by index arrays
into the flat unconstrained vector.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Inference Gym model -> names of the hierarchical components; all others are latent.
HIERARCHICAL_COMPONENTS = {
    "eight_schools": ("log_stddev", "avg_effect"),
    "radon_indiana": ("county_effect_scale", "county_effect_mean", "weight", "log_radon_scale"),
    "radon_minnesota": ("county_effect_scale", "county_effect_mean", "weight", "log_radon_scale"),
    "radon_halfnormal_indiana": ("county_effect_scale", "county_effect_mean", "weight", "log_radon_scale"),
    "item_response": ("mean_student_ability",),
    "stochastic_volatility": ("persistence_of_volatility", "mean_log_volatility", "white_noise_shock_scale"),
    "stochastic_volatility_small": ("persistence_of_volatility", "mean_log_volatility", "white_noise_shock_scale"),
    "log_gaussian_cox": ("length_scale", "amplitude"),
    "brownian_unknown_scales": ("innovation_noise_scale", "observation_noise_scale"),
    "german_credit_sparse": ("global_scale",),
}


def hierarchical_indices(name, gym_model, dim):
    """Indices of the hierarchical and latent coordinates of an Inference Gym target.

    Parameters
    ----------
    name : str
        Catalog name, key of ``HIERARCHICAL_COMPONENTS``.
    gym_model : VectorModel
        The flattened Inference Gym model.
    dim : int
        Dimension of the flat vector.

    Returns
    -------
    hier_idx, latent_idx : sorted int arrays that partition ``range(dim)``.
    """
    structured = gym_model.vector_event_to_structured(jnp.arange(dim, dtype=jnp.float64))
    fields = structured._asdict() if hasattr(structured, "_asdict") else dict(structured)
    wanted = set(HIERARCHICAL_COMPONENTS[name])
    missing = wanted - set(fields)
    if missing:
        raise ValueError(f"{name}: unknown components {sorted(missing)}")
    hier = np.sort(np.concatenate([np.ravel(np.asarray(v)).astype(int)
                                   for k, v in fields.items() if k in wanted]))
    latent = np.setdiff1d(np.arange(dim), hier)
    return hier, latent


class FunnelTarget(NamedTuple):
    """Analytic hierarchical target, native JAX (Neal's funnel and its data variants)."""

    name: str
    dim: int
    logdensity: Callable
    constrain: Callable
    sample_init: Callable
    hier_idx: np.ndarray
    latent_idx: np.ndarray
    ref_mean_x2: np.ndarray
    ref_var_x2: np.ndarray
    details: dict


def _identity(z):
    return z


def neal_funnel(dim, scale_v=3.0, init_std=1.0):
    """Neal's funnel: ``v ~ N(0, scale_v^2)``, ``x_i | v ~ N(0, exp(v))``, ``i = 1..dim-1``.

    Coordinates are ``z = (v, x_1, ..., x_{dim-1})``; ``v`` is hierarchical.
    The scored observable is ``z**2``: ``E v^2 = s^2``, ``Var v^2 = 2 s^4``,
    ``E x^2 = exp(s^2/2)``, ``E x^4 = 3 exp(2 s^2)``.
    """
    s2 = scale_v ** 2

    def logdensity(z):
        v, x = z[0], z[1:]
        return (-0.5 * v ** 2 / s2 - 0.5 * (dim - 1) * v - 0.5 * jnp.sum(x ** 2) * jnp.exp(-v))

    ex2 = np.concatenate([[s2], np.full(dim - 1, np.exp(0.5 * s2))])
    var = np.concatenate([[2 * s2 ** 2], np.full(dim - 1, 3 * np.exp(2 * s2) - np.exp(s2))])
    return FunnelTarget(
        f"funnel_d{dim}", dim, logdensity, _identity,
        lambda key: init_std * jax.random.normal(key, (dim,), jnp.float64),
        np.array([0]), np.arange(1, dim), ex2, var,
        {"scale_v": scale_v, "reference": "analytic"},
    )


def funnel_with_data(dim, noise_std, seed, scale_v=3.0, init_std=1.0, num_observed=None):
    """Neal's funnel with Gaussian data on the latent coordinates.

    Data ``y_i ~ N(x_i, noise_std^2)`` for the first ``num_observed`` latent coordinates,
    generated once from the prior with a fixed seed. The posterior of ``v`` is one-dimensional
    and evaluated by quadrature, which gives exact reference moments of ``v^2`` and ``x_i^2``.
    """
    num_observed = dim - 1 if num_observed is None else num_observed
    rng = np.random.RandomState(seed)
    v_true = rng.normal(0.0, scale_v)
    x_true = rng.normal(0.0, np.exp(0.5 * v_true), size=dim - 1)
    y = x_true[:num_observed] + noise_std * rng.normal(size=num_observed)
    y_j, s2, n2 = jnp.asarray(y), scale_v ** 2, noise_std ** 2

    def logdensity(z):
        v, x = z[0], z[1:]
        prior = -0.5 * v ** 2 / s2 - 0.5 * (dim - 1) * v - 0.5 * jnp.sum(x ** 2) * jnp.exp(-v)
        return prior - 0.5 * jnp.sum((y_j - x[:num_observed]) ** 2) / n2

    ref_mean, ref_var = _funnel_data_reference(y, dim, num_observed, noise_std, scale_v)
    return FunnelTarget(
        f"funnel_data_d{dim}_n{num_observed}_s{noise_std:g}", dim, logdensity, _identity,
        lambda key: init_std * jax.random.normal(key, (dim,), jnp.float64),
        np.array([0]), np.arange(1, dim), ref_mean, ref_var,
        {"scale_v": scale_v, "noise_std": noise_std, "seed": seed, "num_observed": num_observed,
         "v_true": float(v_true), "reference": "quadrature over v"},
    )


def _funnel_data_reference(y, dim, num_observed, noise_std, scale_v):
    """Exact E and Var of the squared coordinates by quadrature over ``v`` (float64).

    Given ``v`` the latent coordinates are independent Gaussian. Observed ones have precision
    ``exp(-v) + 1/noise^2``, mean ``y_i/(noise^2 prec)``; unobserved ones are N(0, exp(v)).
    The marginal of ``v`` is prior times ``prod_i N(y_i; 0, exp(v) + noise^2)``.
    """
    v = np.linspace(-12.0 * scale_v, 12.0 * scale_v, 40001)
    e, n2 = np.exp(v), noise_std ** 2
    log_marg = -0.5 * v ** 2 / scale_v ** 2 - 0.5 * np.sum(np.log(e[:, None] + n2) + y[None, :] ** 2 / (e[:, None] + n2), axis=1)
    weight = np.exp(log_marg - log_marg.max())
    weight /= weight.sum()
    prec = np.exp(-v)[:, None] + 1.0 / n2
    post_var = 1.0 / prec
    post_mean = (y[None, :] / n2) * post_var
    m2_obs = post_var + post_mean ** 2
    m4_obs = 3 * post_var ** 2 + 6 * post_var * post_mean ** 2 + post_mean ** 4
    m2_unobs = e[:, None] * np.ones((1, dim - 1 - num_observed))
    m4_unobs = 3 * e[:, None] ** 2 * np.ones((1, dim - 1 - num_observed))
    m2 = np.concatenate([m2_obs, m2_unobs], axis=1)
    m4 = np.concatenate([m4_obs, m4_unobs], axis=1)
    e_x2 = weight @ m2
    var_x2 = weight @ m4 - e_x2 ** 2
    e_v2 = weight @ v ** 2
    var_v2 = weight @ v ** 4 - e_v2 ** 2
    return np.concatenate([[e_v2], e_x2]), np.concatenate([[var_v2], var_x2])
