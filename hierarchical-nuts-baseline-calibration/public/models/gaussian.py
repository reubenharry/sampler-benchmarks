"""Zero-mean Gaussian target with exact reference second moments."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Gaussian(NamedTuple):
    """Gaussian target N(0, cov) with a Gaussian initialization distribution.

    Attributes
    ----------
    cov : covariance matrix, shape (d, d).
    init_mean : mean of the initialization distribution, shape (d,).
    init_std : standard deviation (isotropic) of the initialization distribution.
    """

    cov: np.ndarray
    init_mean: np.ndarray
    init_std: float

    @property
    def dim(self):
        return self.cov.shape[0]

    def logdensity(self, x):
        precision = jnp.linalg.inv(jnp.asarray(self.cov))
        return -0.5 * x @ precision @ x

    def sample_init(self, key):
        return jnp.asarray(self.init_mean) + self.init_std * jax.random.normal(key, (self.dim,))

    def reference(self):
        """Exact E[x_j^2] and Var[x_j^2]."""
        variance = np.diag(self.cov)
        return variance, 2.0 * variance**2


def diagonal_gaussian(scales, init_std=1.0):
    """Gaussian with independent coordinates of the given standard deviations."""
    scales = np.asarray(scales, dtype=float)
    return Gaussian(np.diag(scales**2), np.zeros_like(scales), init_std)
