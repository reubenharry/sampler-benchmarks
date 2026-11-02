"""Closed-form reference moments agree with numerical quadrature."""
import numpy as np

from models.analytic_references import banana_reference, funnel_reference


def _gauss(scale, n=40):
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return scale * nodes, weights / weights.sum()


def test_banana_moments_by_quadrature():
    x0, w0 = _gauss(10.0)
    z, wz = _gauss(1.0, 20)
    x1 = 0.03 * (x0[:, None] ** 2 - 100.0) + z[None, :]
    weight = w0[:, None] * wz[None, :]
    e2 = np.sum(weight * x1**2)
    e4 = np.sum(weight * x1**4)
    ref_e2_var = banana_reference(2)
    assert np.isclose(ref_e2_var[1][1] + e2**2, e4, rtol=1e-8)
    assert np.isclose(e2, 19.0, rtol=1e-10)


def test_funnel_moments_by_quadrature():
    x0, w0 = _gauss(3.0, 120)
    e2 = np.sum(w0 * np.exp(x0))
    e4 = np.sum(w0 * 3.0 * np.exp(2 * x0))
    _, var = funnel_reference(5)
    assert np.isclose(var[1] + e2**2, e4, rtol=1e-6)
    assert np.isclose(e2, np.exp(4.5), rtol=1e-10)
