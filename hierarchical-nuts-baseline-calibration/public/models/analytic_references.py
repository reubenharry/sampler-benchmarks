"""Exact E[x_j^2] and Var[x_j^2] for targets whose moments are known in closed form."""
import numpy as np


def banana_reference(ndims=2, curvature=0.03):
    """Inference Gym Banana: x0 ~ N(0, 10^2), x1 = curvature * (x0^2 - 100) + N(0, 1), rest N(0, 1)."""
    u2 = 2.0 * 1e4  # E[(x0^2 - 100)^2]
    u4 = 60.0 * 1e8  # E[(x0^2 - 100)^4], chi-square(1) fourth central moment is 60
    e2 = np.ones(ndims)
    e4 = 3.0 * np.ones(ndims)
    e2[0], e4[0] = 100.0, 3.0 * 1e4
    e2[1] = curvature**2 * u2 + 1.0
    e4[1] = curvature**4 * u4 + 6.0 * curvature**2 * u2 + 3.0
    return e2, e4 - e2**2


def funnel_reference(ndims):
    """Inference Gym Neal's funnel: x0 ~ N(0, 3^2), x_i | x0 ~ N(0, exp(x0))."""
    e2 = np.full(ndims, np.exp(4.5))
    e4 = np.full(ndims, 3.0 * np.exp(18.0))
    e2[0], e4[0] = 9.0, 3.0 * 81.0
    return e2, e4 - e2**2


def gaussian_reference(gym_model):
    """Exact E[x^2], Var[x^2] of a Gaussian target from its Inference Gym ground truth."""
    truth = gym_model.sample_transformations["identity"]
    mean = np.asarray(truth.ground_truth_mean, dtype=float)
    variance = np.asarray(truth.ground_truth_standard_deviation, dtype=float) ** 2
    return variance + mean**2, 2.0 * variance**2 + 4.0 * mean**2 * variance
