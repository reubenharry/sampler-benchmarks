"""NUTS baseline calibration with one acceptance rate per problem.

1. Acceptance rate: at the longest grid warmup whose median warmup cost is within the cap, run NUTS at
   0.8, then 0.9, then 0.99 until the median max-error curve crosses the threshold within the stage-1
   horizon; the first success is fixed (0.99 if none succeeds, recorded as failed).
2. Warmup: grid over warmup lengths with median warmup cost within the cap at the fixed rate
   (``--coarse-chains`` chains, stage-1 horizon); choose the smallest crossing count.
3. Final: ``--chains`` (100) chains with a fresh seed at the chosen setting; crossing counts at several
   thresholds are recorded so that the threshold can be fixed after the pilot.
Appends one JSON line per problem to the output file.

Usage: python tools/calibrate_nuts.py OUT.jsonl PROBLEM [PROBLEM ...]
"""
import argparse, json, sys, time, zlib
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, "public")
from baseline import nuts_baseline as nb
from evaluator.evaluate import Problem
from models.catalog import build_hierarchical_target

THRESHOLDS = (0.01, 0.02, 0.05, 0.1)
WARMUP_GRID = (100, 500, 1000, 2000)
WARMUP_COST_CAP = 100000
RATES = (0.8, 0.9, 0.99)


def make_problem(name):
    t = build_hierarchical_target(name)
    return Problem(name, t.dim, t.logdensity, t.constrain, t.sample_init, t.ref_mean, t.ref_var, np.nan,
                   t.hier_idx, t.latent_idx)


def counts(run, horizon):
    return {str(th): nb.crossing_count(run, horizon, th) for th in THRESHOLDS}


def usable_warmups(problem, key, rate, chains):
    """Grid warmups whose median warmup cost is within the cap (longest first), with their costs."""
    out = {}
    for warmup in sorted(WARMUP_GRID, reverse=True):
        cost = float(np.median(nb.warmup_costs(problem, key, chains, warmup, rate)))
        out[warmup] = cost
        if cost <= WARMUP_COST_CAP:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("problems", nargs="+")
    parser.add_argument("--chains", type=int, default=100)
    parser.add_argument("--coarse-chains", type=int, default=10)
    parser.add_argument("--coarse-horizon", type=int, default=1000000)
    parser.add_argument("--horizon", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-seed", type=int, default=1)
    parser.add_argument("--threshold", default="0.01")
    args = parser.parse_args()
    for name in args.problems:
        problem = make_problem(name)
        folded = lambda seed: jax.random.fold_in(jax.random.key(seed), zlib.crc32(name.encode()) % (2**31))
        key, final_key = folded(args.seed), folded(args.final_seed)
        record = {"problem": name, "dim": problem.dim, "seed": args.seed, "final_seed": args.final_seed,
                  "coarse_horizon": args.coarse_horizon, "horizon": args.horizon, "grid": list(WARMUP_GRID),
                  "coarse_chains": args.coarse_chains, "chains": args.chains, "rate_search": [], "grid_runs": {}}
        t0 = time.time()

        rate, rate_ok = RATES[-1], False
        for candidate in RATES:
            costs = usable_warmups(problem, jax.random.fold_in(key, 0), candidate, args.coarse_chains)
            longest = max((w for w, c in costs.items() if c <= WARMUP_COST_CAP), default=None)
            entry = {"rate": candidate, "warmup_costs_probed": costs, "longest_usable": longest}
            if longest is not None:
                run = nb.run_nuts(problem, jax.random.fold_in(key, 1), args.coarse_chains, longest, candidate,
                                  args.coarse_horizon)
                entry["counts"] = counts(run, args.coarse_horizon)
                entry["success"] = bool(np.isfinite(entry["counts"][args.threshold]))
            else:
                entry["success"] = False
            record["rate_search"].append(entry)
            print(name, "rate search", entry, flush=True)
            if entry["success"]:
                rate, rate_ok = candidate, True
                break
        record["rate"], record["rate_search_succeeded"] = rate, rate_ok

        usable = {}
        for warmup in WARMUP_GRID:
            cost = float(np.median(nb.warmup_costs(problem, jax.random.fold_in(key, 0), args.coarse_chains, warmup, rate)))
            if cost > WARMUP_COST_CAP:
                record["grid_runs"][str(warmup)] = {"skipped": f"median warmup cost {cost:.0f} > cap"}
                continue
            run = nb.run_nuts(problem, jax.random.fold_in(key, 2), args.coarse_chains, warmup, rate, args.coarse_horizon)
            record["grid_runs"][str(warmup)] = {"median_warmup_cost": cost, "counts": counts(run, args.coarse_horizon)}
            usable[warmup] = record["grid_runs"][str(warmup)]["counts"]
            print(name, rate, warmup, record["grid_runs"][str(warmup)], flush=True)
        best = None
        for th in (args.threshold, "0.02", "0.05", "0.1"):  # fall back to looser thresholds if none crosses
            finite = {w: c[th] for w, c in usable.items() if np.isfinite(c[th])}
            if finite:
                best, record["selection_threshold"] = min(finite, key=finite.get), th
                break
        record["chosen_warmup"] = best
        if best is not None:
            run = nb.run_nuts(problem, final_key, args.chains, best, rate, args.horizon)
            record["final"] = {"median_warmup_cost": float(np.median(run["warmup_cost"])),
                               "counts": counts(run, args.horizon),
                               "counts_mean_error": {str(th): nb.crossing_count(run, args.horizon, th, "error_mean")
                                                     for th in THRESHOLDS}}
            print(name, "FINAL", record["final"], flush=True)
        record["seconds"] = time.time() - t0
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


if __name__ == "__main__":
    main()
