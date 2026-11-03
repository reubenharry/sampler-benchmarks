"""Check that sharding chains over devices gives the same per-chain results: ``python tools/check_chain_map.py``."""
import sys

import jax
import numpy as np

sys.path.insert(0, "public")
sys.path.insert(0, "tools")
jax.config.update("jax_enable_x64", True)

from baseline import nuts_baseline as nb
from calibrate_nuts import make_problem


def main():
    problem = make_problem("eight_schools")
    run = nb.run_nuts(problem, jax.random.key(0), 8, 100, 0.8, 20000, stride=10)
    print(f"devices={jax.local_device_count()} sum_error_max={np.nansum(run['error_max']):.12e} "
          f"sum_cost={run['total_cost'].sum():.0f} warmup_cost_sum={run['warmup_cost'].sum():.0f}")


if __name__ == "__main__":
    main()
