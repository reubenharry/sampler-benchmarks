"""Lattice field theory from sampler-benchmarks: scalar phi^4.

The log-density is identical to ``sampler_evaluation.models.phi4``. The constrained
summary ``x`` is the momentum-space power spectrum; as for every target the
scored observable is ``x**2``.
"""
import jax
import jax.numpy as jnp

from models.inference_gym_models import Target


def unreduce_lam(reduced_lam, side):
    """Coupling at reduced distance from the critical point (arXiv:2207.00283, Fig. 3)."""
    return 4.25 * (reduced_lam / side + 1.0)


def phi4(side, lam):
    """Two-dimensional lattice phi^4 theory with periodic boundaries and ``m^2 = -4``."""
    dim = side**2

    def logdensity(x):
        phi = x.reshape(side, side)
        neighbours = (jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1))
        return -jnp.sum(lam * phi**4 - phi * neighbours)

    def constrain(x):
        return (jnp.square(jnp.abs(jnp.fft.fft2(x.reshape(side, side)))) / side**2).reshape(dim)

    return Target(f"phi4_L{side}_lam{str(lam)[:5]}", dim, logdensity, constrain,
                  lambda key: jax.random.normal(key, (dim,), jnp.float64), "float64", None)


# name -> constructor; lattice sizes, couplings and inverse temperatures of the benchmark problems
LATTICE_CATALOG = {
    "phi4_L8_lam5.198": lambda: phi4(8, 5.198),
    "phi4_L16_lam4.344": lambda: phi4(16, 4.344),
}
