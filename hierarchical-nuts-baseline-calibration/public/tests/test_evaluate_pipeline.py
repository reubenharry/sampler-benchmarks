"""End-to-end evaluator checks on a small Gaussian problem."""
import jax
import numpy as np

from baseline.mclmc_adapter import make_baseline
from evaluator.evaluate import Problem, evaluate_submission, production_cap
from examples import submission as minimal
from models.gaussian import diagonal_gaussian

jax.config.update("jax_enable_x64", True)


def _problem(n_base=2000.0):
    model = diagonal_gaussian(np.geomspace(0.5, 2.0, 6), init_std=1.0)
    mean, var = model.reference()
    return Problem("gauss6", 6, model.logdensity, lambda z: z, model.sample_init, mean, var, n_base,
                   np.array([0, 1]), np.arange(2, 6))


def test_baseline_submission_converges_and_scores_finite():
    out = evaluate_submission(make_baseline(1000), [_problem()], bootstrap=20)
    rec = out["per_problem"][0]
    assert rec["valid_chains"] == 100 and np.isfinite(rec["total_n"])
    assert rec["burn_in_used"] == 667 and out["disqualified"] is None
    assert np.isfinite(out["F"]) and out["F"] > 0
    assert "bootstrap" in out


def test_default_submission_may_fail_to_converge_and_scores_zero():
    out = evaluate_submission(minimal, [_problem(n_base=30.0)], bootstrap=0)  # tiny cap: cannot converge
    assert out["F"] == 0.0 and out["excluded_problems"] == 1


def test_overspending_schedule_is_disqualified():
    class Greedy:
        phase_fractions = staticmethod(lambda config: (0.7, 0.7))
        init = staticmethod(minimal.init)
        step_phase = staticmethod(minimal.step_phase)
        transition = staticmethod(minimal.transition)
        finalize = staticmethod(minimal.finalize)

    out = evaluate_submission(Greedy, [_problem()], bootstrap=0)
    assert out["F"] == 0.0 and out["disqualified"] is not None


def test_production_cap_is_five_times_baseline():
    assert production_cap(1888.0) == 5664 and production_cap(1888.0, 5.0) == 9440 and production_cap(1e9) == 500000
