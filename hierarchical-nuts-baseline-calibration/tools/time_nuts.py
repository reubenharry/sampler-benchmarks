"""Wall-clock timing of NUTS runs: ``python tools/time_nuts.py PROBLEM CHAINS WARMUP HORIZON [RATE]``."""
import sys
import time

import jax
import numpy as np

sys.path.insert(0, "public")
sys.path.insert(0, "tools")
jax.config.update("jax_enable_x64", True)

from baseline import nuts_baseline as nb
from calibrate_nuts import make_problem


def main():
    name, chains, warmup, horizon = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
    problem = make_problem(name)
    start = time.time()
    run = nb.run_nuts(problem, jax.random.key(0), chains, warmup, rate, horizon)
    seconds = time.time() - start
    cost = run["total_cost"]
    print(f"{name} chains={chains} warmup={warmup} rate={rate} horizon={horizon} seconds={seconds:.1f} "
          f"warmup_cost={np.median(run['warmup_cost']):.0f} max_total_cost={cost.max():.0f} "
          f"records={cost.shape[1]} crossing_0.01={nb.crossing_count(run, horizon, 0.01)}", flush=True)


if __name__ == "__main__":
    main()
