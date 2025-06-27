This repo is a wrapper on top of blackjax. For the full suite of evaluations, users will need to locally install the following fork of blackjax:

TODO

## How to reproduce results

You will need to locally install three packages: `sampler-comparison` (inside this repo), `sampler-evaluation` (inside this repo) and `blackjax` (local version, linked separately, on branch `working_branch`). Use pip to install `jax`.

### Main results

The results are generated for each model separately, and can be found in `sampler-comparison/results/`:
- `ICG`
- `rosenbrock_36d`
- `Stochastic_Volatility_MAMS_Paper`
- `vector_brownian_motion_unknown_scales_missing_middle_observations`
- `vector_german_credit_numeric_sparse_logistic_regression`
- `vector_synthetic_item_response_theory`

In each, `main.py` generates results for table 2 and 3, excepting the grid search results, which are produced by `grid_search.py`. An example of a results table that gets produced by `main.py` is `results/ICG/nuts_ICG.csv.

In most cases, an inference gym model is used. For `Rosenbrock_36D()`, we provide our own product of 18 2D Rosenbrocks. For `stochastic_volatility_mams_paper`, we adapt the NumPyro Stochastic Volatility model, which differs from inference gym in number of dimensions and presents a harder inference problem (more steps required for convergence by a factor of 10).  For `banana_mams_paper`, our implementation aligns with inference gym, and is present to avoid [a now resolved bug incurred when using 64 bit precision](https://github.com/tensorflow/probability/issues/1993). For `IllConditionedGaussian`, we again provide our own variant of a Gaussian, which is unrotated. For Neal's Funnel, we use our own implementation, which should be similar or the same to the inference gym version.

For the models marked with "MAMS", please disregard the ground truths of the first moments (e.g. `{"identity": SampleTransformation(ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf)}`), which are set to infinity. In any case, we only inspect the second moments.

Ground truth values for the expectation of $x^2$ are generated in the `sampler-evaluation` package, in `sampler_evaluation/models/data/estimate_expectations.py`, with the exception of `stochastic_volatility_mams_paper`, where a pre-existing ground truth was used, and models where the analytic form of the expectation is known.

