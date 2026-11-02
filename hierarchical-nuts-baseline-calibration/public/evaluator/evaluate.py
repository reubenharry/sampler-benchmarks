"""Evaluate a frozen submission on a suite of problems.

The same functions score public and private problems; only the ``Problem``
records differ (targets, references and baseline counts are supplied as data).
"""
import time
import zlib
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from evaluator import metrics
from evaluator.interface import PublicConfig, identity_transform, is_valid_result, phase_lengths
from evaluator.runner import run_burn_in, run_production_errors

MAX_BURN_IN = 20000
PRODUCTION_FACTOR = 3.0  # first production horizon: floor(3 * n_base)
EXTENDED_FACTOR = 5.0  # used if the median curve has not converged: floor(5 * n_base)
PRODUCTION_HARD_CAP = 500000
NUM_CHAINS = 100


class Problem(NamedTuple):
    """One benchmark problem.

    Attributes
    ----------
    name : identifier used for seeding and reporting.
    dim : dimension of the unconstrained vector.
    logdensity : callable, ``z`` of shape (dim,) to scalar.
    constrain : callable, ``z`` to the vector whose squares are scored.
    sample_init : callable, key to an initial point.
    ref_mean, ref_var : reference E[x_j^2] and Var[x_j^2].
    n_base : baseline total gradient evaluations (burn-in included) to convergence.
    hier_idx, latent_idx : index arrays of the hierarchical and latent coordinates.
    """

    name: str
    dim: int
    logdensity: Callable
    constrain: Callable
    sample_init: Callable
    ref_mean: np.ndarray
    ref_var: np.ndarray
    n_base: float
    hier_idx: np.ndarray
    latent_idx: np.ndarray


def production_cap(n_base, factor=PRODUCTION_FACTOR):
    """Maximum production evaluations ``H_i = min(floor(factor * n_base), 500000)``."""
    return int(min(np.floor(factor * n_base), PRODUCTION_HARD_CAP))


def _problem_key(seed, name):
    return jax.random.fold_in(jax.random.key(seed), zlib.crc32(name.encode()) % (2**31))


def _chain_validity(results):
    n_chains = results.position.shape[0]
    valid = []
    for i in range(n_chains):
        chain = jax.tree.map(lambda a: a[i], results)
        valid.append(is_valid_result(chain))
    return np.array(valid)


def _transform_validity(transform, problem, results):
    """Per chain: the transform at the starting point is finite with a finite log-determinant."""
    def finite(result):
        z, logdet = transform(result.transform_params, result.position)
        return jnp.all(jnp.isfinite(z)) & jnp.isfinite(logdet) & (z.shape == (problem.dim,))
    return np.asarray(jax.vmap(finite)(results))


def evaluate_problem(submission, problem, seed=0, num_chains=NUM_CHAINS, horizon=None):
    """Run the submission on one problem; return counts, curves and failure information."""
    key = _problem_key(seed, problem.name)
    init_key, burn_key, prod_key = jax.random.split(key, 3)
    x0 = jax.vmap(problem.sample_init)(jax.random.split(init_key, num_chains))
    horizon = production_cap(problem.n_base) if horizon is None else horizon
    config = PublicConfig(dim=problem.dim, budget=MAX_BURN_IN, hier_idx=tuple(problem.hier_idx),
                          latent_idx=tuple(problem.latent_idx))
    lengths = phase_lengths(submission.phase_fractions(config), MAX_BURN_IN)
    burn_in_used = int(sum(lengths))

    t0 = time.time()
    burn = jax.jit(jax.vmap(lambda x, k: run_burn_in(submission, problem.logdensity, x, k, MAX_BURN_IN,
                                                          problem.hier_idx, problem.latent_idx)[0]))
    results = jax.block_until_ready(burn(x0, jax.random.split(burn_key, num_chains)))
    burn_seconds = time.time() - t0

    transform = getattr(submission, "transform", identity_transform)
    valid = _chain_validity(results) & _transform_validity(transform, problem, results)
    observable = lambda z: jnp.square(problem.constrain(z))
    ref_mean, ref_var = jnp.asarray(problem.ref_mean), jnp.asarray(problem.ref_var)

    def run_horizon(horizon):
        def production(result, chain_key):
            return run_production_errors(result, transform, problem.logdensity, chain_key, horizon,
                                         observable, ref_mean, ref_var)

        costs, _, e_max = jax.block_until_ready(
            jax.jit(jax.vmap(production))(results, jax.random.split(prod_key, num_chains)))
        costs, e_max = np.asarray(costs)[0], np.asarray(e_max)
        e_max = np.where(valid[:, None], e_max, np.inf)  # invalid chains never converge
        return costs, e_max, metrics.median_curve_crossing(e_max, costs)

    t0 = time.time()
    horizon = production_cap(problem.n_base) if horizon is None else horizon
    costs, e_max, crossing = run_horizon(horizon)
    extended = production_cap(problem.n_base, EXTENDED_FACTOR)
    if not np.isfinite(crossing) and extended > horizon:  # not converged: extend to floor(5 n), same seeds
        horizon = extended
        costs, e_max, crossing = run_horizon(horizon)
    production_seconds = time.time() - t0
    return {
        "name": problem.name, "burn_in_used": burn_in_used, "phase_lengths": list(lengths),
        "production_cap": horizon, "valid_chains": int(valid.sum()),
        "production_n": float(crossing), "total_n": metrics.total_count(burn_in_used, crossing),
        "n_base": float(problem.n_base), "error_max": e_max, "costs": costs,
        "final_error_median": float(np.nanmedian(np.where(np.isfinite(e_max[:, -1]), e_max[:, -1], np.nan)))
        if valid.any() else float("nan"),
        "seconds": {"burn_in": burn_seconds, "production": production_seconds},
    }


def bootstrap_scores(per_problem, num_resamples=200, seed=0):
    """Bootstrap over chains (infinite values retained); returns F quantiles and the fraction of zero scores."""
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(num_resamples):
        totals = []
        for rec in per_problem:
            idx = rng.randint(0, rec["error_max"].shape[0], rec["error_max"].shape[0])
            crossing = metrics.median_curve_crossing(rec["error_max"][idx], rec["costs"])
            totals.append(metrics.total_count(rec["burn_in_used"], crossing))
        samples.append(metrics.score([r["n_base"] for r in per_problem], totals)["F"])
    samples = np.array(samples)
    return {"F_quantiles_16_50_84": [float(q) for q in np.percentile(samples, [16, 50, 84])],
            "fraction_zero": float(np.mean(samples == 0.0))}


def evaluate_submission(submission, problems, seed=0, num_chains=NUM_CHAINS, bootstrap=200):
    """Evaluate on all problems and aggregate the primary and finite-problem scores.

    Budget violations (phase fractions that would exceed the burn-in cap) are
    reported as disqualification with official score 0.
    """
    per_problem, disqualified = [], None
    for problem in problems:
        try:
            per_problem.append(evaluate_problem(submission, problem, seed, num_chains))
        except ValueError as error:  # invalid phase schedule: attempted budget violation
            disqualified = f"{problem.name}: {error}"
            break
    if disqualified:
        return {"disqualified": disqualified, "F": 0.0, "per_problem": per_problem}
    scores = metrics.score([r["n_base"] for r in per_problem], [r["total_n"] for r in per_problem])
    out = {"disqualified": None, **scores, "per_problem": per_problem, "seed": seed}
    if bootstrap:
        out["bootstrap"] = bootstrap_scores(per_problem, bootstrap)
    return out
