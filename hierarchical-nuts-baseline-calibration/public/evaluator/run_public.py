"""Evaluate a submission on the public development problems.

Usage:
    python -m evaluator.run_public SUBMISSION.py [--seed 0] [--problems name ...] [--out result.json] [--bootstrap 200]

Requires ``baseline/calibration.json`` (baseline counts of the public problems).
"""
import argparse
import importlib.util
import json
import os

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from evaluator.compliance import check_file  # noqa: E402
from evaluator.evaluate import Problem, evaluate_submission  # noqa: E402
from models.catalog import build_hierarchical_target  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def public_problems(names=None):
    """Build the public ``Problem`` records from the catalog, references and calibration table."""
    with open(os.path.join(ROOT, "baseline", "calibration.json")) as handle:
        calibration = json.load(handle)
    names = names or list(calibration)
    problems = []
    for name in names:
        target = build_hierarchical_target(name)
        problems.append(Problem(name, target.dim, target.logdensity, target.constrain, target.sample_init,
                                target.ref_mean, target.ref_var, calibration[name]["n_base"],
                                target.hier_idx, target.latent_idx))
    return problems


def load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problems", nargs="*")
    parser.add_argument("--out", default="result.json")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    violations = check_file(args.submission)
    if violations:
        for line, message in violations:
            print(f"{args.submission}:{line}: {message}")
        raise SystemExit("compliance screening failed")
    result = evaluate_submission(load_submission(args.submission), public_problems(args.problems), args.seed, bootstrap=args.bootstrap)
    for record in result["per_problem"]:
        record["error_max"], record["costs"] = None, None
        print(f"{record['name']:30s} total n = {record['total_n']:10.0f} (burn-in {record['burn_in_used']}, baseline {record['n_base']:.0f})")
    print({k: v for k, v in result.items() if k != "per_problem"})
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, default=float)


if __name__ == "__main__":
    main()
