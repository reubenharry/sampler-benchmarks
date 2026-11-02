"""Inference Gym targets in unconstrained coordinates, as used by sampler-benchmarks.

The log-density is ``log p(bijector(z)) + log|det J(z)|`` for the flattened
model, and scored observables are the squares of the constrained coordinates
``bijector(z)``. Requires ``tfp-nightly`` and ``inference_gym`` (see
``requirements``); the three lines below are the TFP/JAX compatibility patch
from sampler-benchmarks (``tfp_jax_patch.py``).
"""
from typing import Callable, NamedTuple

import jax
import jax.interpreters.xla as _xla
import jax.numpy as jnp

if not hasattr(_xla, "pytype_aval_mappings"):
    _xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import inference_gym.using_jax as gym  # noqa: E402
import inference_gym.dynamic.backend_jax.internal.data as _gym_data  # noqa: E402

from models import tfds_shim  # noqa: E402

_gym_data._tfds = lambda: tfds_shim

# name -> (Inference Gym class, keyword arguments)
CATALOG = {
    "banana": ("Banana", {}),
    "brownian_unknown_scales": ("BrownianMotionUnknownScalesMissingMiddleObservations", {}),
    "brownian_missing_middle": ("BrownianMotionMissingMiddleObservations", {}),
    "eight_schools": ("EightSchools", {}),
    "german_credit_logistic": ("GermanCreditNumericLogisticRegression", {}),
    "german_credit_probit": ("GermanCreditNumericProbitRegression", {}),
    "german_credit_sparse": ("GermanCreditNumericSparseLogisticRegression", {}),
    "item_response": ("SyntheticItemResponseTheory", {}),
    "log_gaussian_cox": ("SyntheticLogGaussianCoxProcess", {}),
    "quartic": ("NonIdentifiableQuarticMeasurementModel", {}),
    "radon_indiana": ("RadonContextualEffectsIndiana", {}),
    "radon_minnesota": ("RadonContextualEffectsMinnesota", {}),
    "radon_halfnormal_indiana": ("RadonContextualEffectsHalfNormalIndiana", {}),
    "stochastic_volatility": ("VectorizedStochasticVolatilityLogSP500", {}),
    "stochastic_volatility_small": ("VectorizedStochasticVolatilityLogSP500Small", {}),
    "dirichlet": ("Dirichlet", {}),
    "ill_conditioned_gaussian": ("IllConditionedGaussian", {}),
}


class Target(NamedTuple):
    """A sampling problem in unconstrained coordinates.

    Attributes
    ----------
    name : catalog name.
    dim : dimension of the unconstrained vector.
    logdensity : callable, unconstrained ``z`` of shape (dim,) to scalar.
    constrain : callable, ``z`` to the flattened constrained sample ``x``.
    sample_init : callable, key to an initial ``z`` (standard normal).
    dtype : floating-point dtype of the underlying Inference Gym model.
    gym_model : the flattened Inference Gym model (for diagnostics only).
    """

    name: str
    dim: int
    logdensity: Callable
    constrain: Callable
    sample_init: Callable
    dtype: str
    gym_model: object


def _construct(class_name, kwargs):
    cls = getattr(gym.targets, class_name)
    try:
        return cls(dtype=jnp.float64, **kwargs), "float64"
    except TypeError:
        return cls(**kwargs), "model default (no dtype argument)"


def build_target(name):
    """Build a catalog target with its catalog initialization distribution."""
    from models.init_distributions import with_init
    return with_init(_build_target(name))


def _build_target(name):
    """Build a catalog target.

    Parameters
    ----------
    name : str
        Key of ``CATALOG``.

    Returns
    -------
    Target
    """
    from models import lattice
    if name in lattice.LATTICE_CATALOG:
        return lattice.LATTICE_CATALOG[name]()
    class_name, kwargs = CATALOG[name]
    model, dtype = _construct(class_name, kwargs)
    return target_from_gym_model(name, model, dtype)


def target_from_gym_model(name, model, dtype="float64"):
    """Wrap an Inference Gym model (constructor arguments carry the data) as a ``Target``."""
    flat = gym.targets.VectorModel(model, flatten_sample_transformations=True)
    bijector = flat.default_event_space_bijector
    dim = int(jnp.prod(jnp.asarray(flat.event_shape)))
    transform = flat.sample_transformations["identity"].fn

    def logdensity(z):
        x = bijector(z)
        value = flat.unnormalized_log_prob(x) + bijector.forward_log_det_jacobian(z, event_ndims=1)
        return jnp.asarray(value, jnp.float64)

    def constrain(z):
        return jnp.asarray(transform(bijector(z)), jnp.float64)

    def sample_init(key):
        return jax.random.normal(key, (dim,), jnp.float64)

    return Target(name, dim, logdensity, constrain, sample_init, dtype, flat)
