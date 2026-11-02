"""Public schemas of the burn-in interface and validation of submitted outputs."""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from evaluator.mclmc_stepper import Beta

REQUIRED_CARRY_KEYS = ("query_position", "production_position", "beta")


class PublicConfig(NamedTuple):
    """Public, static configuration passed to ``init``, ``phase_fractions`` and ``finalize``.

    Attributes
    ----------
    dim : problem dimension.
    budget : maximum number of burn-in gradient evaluations for this chain.
    hier_idx : sorted indices of the hierarchical coordinates (few: hyper- and global parameters).
    latent_idx : sorted indices of the latent coordinates; together with ``hier_idx`` they partition ``range(dim)``.
    """

    dim: int
    budget: int
    hier_idx: Any = ()
    latent_idx: Any = ()


class Observation(NamedTuple):
    """Numerical observation at the query point of one active iteration.

    Attributes
    ----------
    position : point that was evaluated, shape (d,).
    logp : log-density value at ``position``.
    grad : gradient of the log-density at ``position``, shape (d,).
    global_index : evaluation index within the whole burn-in, starting at 0.
    phase_index : evaluation index within the current phase, starting at 0.
    key : random key for this iteration.
    """

    position: jax.Array
    logp: jax.Array
    grad: jax.Array
    global_index: jax.Array
    phase_index: jax.Array
    key: jax.Array


class PhaseHistory(NamedTuple):
    """Observations of one completed phase, length ``n`` equal to the phase length.

    Every entry is valid (no padding). An empty phase has leading dimension 0.
    """

    position: jax.Array
    logp: jax.Array
    grad: jax.Array


class PhaseInfo(NamedTuple):
    """Static description of the phase that just ended."""

    phase: int
    length: int
    num_phases: int


class Result(NamedTuple):
    """Output of ``finalize``.

    Attributes
    ----------
    position : production starting point ``w0`` in the transformed coordinates, shape (d,).
    beta : MCLMC parameters in the transformed coordinates.
    transform_params : pytree of float arrays passed to the module-level ``transform`` of the
        submission (empty for the identity). Production samples ``w`` of the fixed MCLMC target
        ``log p(z(w)) + logdet(w)`` with ``z, logdet = transform(transform_params, w)``.
    """

    position: jax.Array
    beta: Beta
    transform_params: Any = ()


def identity_transform(params, w):
    """Default ``transform``: ``z = w`` with zero log-determinant."""
    return w, jnp.zeros((), w.dtype)


def phase_lengths(fractions, budget):
    """Allocate ``budget`` evaluations over phases.

    Lengths are the differences of ``floor(cumsum(fractions) * budget)``, so they
    are non-negative integers whose sum never exceeds ``budget``.

    Parameters
    ----------
    fractions : sequence of float
        Non-negative phase fractions with sum at most 1.
    budget : int
        Total burn-in evaluations ``B_i``.

    Returns
    -------
    tuple of int
    """
    fractions = [float(f) for f in fractions]
    if len(fractions) == 0 or len(fractions) > MAX_PHASES:
        raise ValueError(f"number of phases must be in 1..{MAX_PHASES}")
    if any((not np.isfinite(f)) or f < 0 for f in fractions):
        raise ValueError("phase fractions must be finite and non-negative")
    if sum(fractions) > 1.0 + 1e-12:
        raise ValueError("phase fractions must sum to at most 1")
    edges = np.floor(np.cumsum(fractions) * budget + 1e-9).astype(int)
    edges = np.minimum(edges, budget)
    return tuple(int(n) for n in np.diff(np.concatenate([[0], edges])))


MAX_PHASES = 8


def check_carry(carry, dim):
    """Raise ``ValueError`` unless the carry has the required standard fields."""
    if not isinstance(carry, dict):
        raise ValueError("carry must be a dict")
    for key in REQUIRED_CARRY_KEYS:
        if key not in carry:
            raise ValueError(f"carry is missing required field '{key}'")
    for key in ("query_position", "production_position"):
        if carry[key].shape != (dim,):
            raise ValueError(f"carry['{key}'] must have shape ({dim},)")


def is_valid_result(result):
    """Return True if position and beta are finite and in the valid range.

    Valid: finite position of the right size, finite positive ``L`` and
    ``step_size``, finite positive diagonal inverse mass matrix, finite transform parameters.
    """
    position = np.asarray(result.position)
    beta = result.beta
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(result.transform_params)):
        return False
    inverse_mass = np.asarray(beta.inverse_mass_matrix)
    L, step_size = float(beta.L), float(beta.step_size)
    return bool(
        np.all(np.isfinite(position))
        and inverse_mass.shape == position.shape
        and np.all(np.isfinite(inverse_mass)) and np.all(inverse_mass > 0)
        and np.isfinite(L) and L > 0 and np.isfinite(step_size) and step_size > 0
    )
