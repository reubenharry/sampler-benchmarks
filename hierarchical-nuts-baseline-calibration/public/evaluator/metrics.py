"""Per-chain error, first crossings, medians and scores."""
import numpy as np

THRESHOLD = 0.01
AGGREGATIONS = {"mean": np.mean, "max": np.max}


def error_curve(positions, ref_mean_x2, ref_var_x2, aggregation="max"):
    """Coordinate aggregate of the normalized squared error of the second moment.

    Parameters
    ----------
    positions : array, shape (n, d)
        Production samples in order.
    ref_mean_x2, ref_var_x2 : arrays, shape (d,)
        Reference expectation and variance of ``x_j**2``.
    aggregation : {"mean", "max"}
        Aggregate over coordinates (default: maximum bias).

    Returns
    -------
    array, shape (n,)
        Error using the first ``k`` samples, for ``k = 1..n``.
    """
    positions = np.asarray(positions)
    counts = np.arange(1, positions.shape[0] + 1)[:, None]
    running_mean = np.cumsum(positions**2, axis=0) / counts
    errors = (running_mean - ref_mean_x2) ** 2 / ref_var_x2
    return AGGREGATIONS[aggregation](errors, axis=1)


def first_crossing(errors, costs, threshold=THRESHOLD):
    """Cost at the first recorded prefix with error strictly below ``threshold``, else inf.

    NaN errors never count as a crossing.
    """
    errors = np.asarray(errors)
    below = np.nonzero(errors < threshold)[0]
    return float(np.asarray(costs)[below[0]]) if below.size else np.inf


GRID_POINTS = 4000


def median_curve_crossing(error_curves, costs, threshold=THRESHOLD):
    """Cost at the first prefix where the median-over-chains error is strictly below ``threshold``.

    The median over chains is taken at every cost first; NaN errors count as
    infinitely large, so failed chains cannot lower the median. Costs may be common to all
    chains (1-D) or chain-specific (2-D, e.g. NUTS whose transitions cost a variable number of
    gradients). In the second case each chain's error is a step function of its own total cost
    (infinite before its first recorded prefix) and the median is taken on a common grid of
    ``GRID_POINTS`` costs, so the crossing is resolved to that grid.

    Parameters
    ----------
    error_curves : array, shape (chains, n)
        Error of every chain after each recorded production prefix.
    costs : array, shape (n,) or (chains, n)
        Cumulative gradient evaluations of each prefix.

    Returns
    -------
    float
        Crossing cost, or ``inf`` if the median curve never goes below the threshold.
    """
    curves = np.where(np.isnan(error_curves), np.inf, error_curves)
    costs = np.asarray(costs, dtype=float)
    if costs.ndim == 1:
        return first_crossing(np.median(curves, axis=0), costs, threshold)
    grid = np.linspace(costs.min(), costs.max(), GRID_POINTS)
    on_grid = np.full((curves.shape[0], GRID_POINTS), np.inf)
    for chain in range(curves.shape[0]):
        last = np.searchsorted(costs[chain], grid, side="right") - 1
        on_grid[chain] = np.where(last >= 0, curves[chain, np.maximum(last, 0)], np.inf)
    return first_crossing(np.median(on_grid, axis=0), grid, threshold)


def total_count(burn_in_evaluations, production_count):
    """Total gradient evaluations to convergence: burn-in actually scheduled plus production."""
    return float(burn_in_evaluations) + float(production_count)


def problem_count(chain_counts):
    """Ordinary median of chain counts, retaining infinities."""
    return float(np.median(np.asarray(chain_counts, dtype=float)))


def score(base_counts, candidate_counts):
    """Primary and finite-problem scores.

    Parameters
    ----------
    base_counts, candidate_counts : sequences of float
        Per-problem medians ``n_i``.

    Returns
    -------
    dict
        ``F`` (0 if any candidate count is infinite), ``F_finite`` (None if no
        finite problem) and ``excluded_fraction``.
    """
    base = np.asarray(base_counts, dtype=float)
    cand = np.asarray(candidate_counts, dtype=float)
    finite = np.isfinite(cand)
    log_ratio = np.log(base) - np.log(cand)
    F = float(np.exp(np.mean(log_ratio))) if finite.all() else 0.0
    F_finite = float(np.exp(np.mean(log_ratio[finite]))) if finite.any() else None
    return {
        "F": F,
        "F_finite": F_finite,
        "excluded_problems": int((~finite).sum()),
        "excluded_fraction": float((~finite).mean()),
    }
