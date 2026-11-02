# NUTS baseline calibration for hierarchical targets

BlackJAX 1.7.1 NUTS (window adaptation, diagonal mass matrix) on hierarchical Bayesian targets.
Gradient evaluations are counted per chain (warmup plus production); the median over chains of the
max-over-coordinates normalised squared error of `x^2` is compared with a threshold of 0.01.

Run from this directory (scripts add `public/` to the path); environment pins in `public/requirements.txt`.

```
python tools/calibrate_nuts.py OUT.jsonl PROBLEM [PROBLEM ...]
python tools/time_nuts.py PROBLEM CHAINS WARMUP HORIZON [RATE]
python -m pytest public/tests/test_hierarchical_and_transform.py
```

`calibrate_nuts.py` per problem: (1) rate search (0.8, 0.9, 0.99) at the longest warmup in {100, 500, 1000, 2000}
whose median warmup cost is within `--warmup-cap`-equivalent constant `WARMUP_COST_CAP` (100000 gradients),
10 chains, stage-1 horizon 1M; (2) warmup grid at the fixed rate; (3) final 100-chain run on a new seed, horizon 2M.
One JSON line per problem is appended to `OUT.jsonl`.
Problems: `public/models/catalog.py::available_names`.
