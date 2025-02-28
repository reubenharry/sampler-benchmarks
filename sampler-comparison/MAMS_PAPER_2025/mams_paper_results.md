# [MAMS Paper]() Result generation

This document outlines the steps to generate the results for the MAMS paper, which was performed using the `sampler-comparison` package on the branch `research` of the `sampler-benchmarks` repository in which `sampler-comparison` is contained.

## Table 1

These results were generated with `sampler_comparison/experiments/experiment.py`.

For the paper, these results were run individually, with a varying number of steps for each model (as long as the sampler converges, the number of steps does not matter), and with a varying number of chains (512 chains for `Banana_MAMS`).

### Notes on the set of models

The model set is:

```python
models={
            "Banana_MAMS": banana_mams_paper,
            "Gaussian_100D": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
            "Rosenbrock_36D": Rosenbrock_36D(),
            "Neals_Funnel_MAMS": neals_funnel_mams_paper,
            "Brownian_Motion": brownian_motion(),
            "German_Credit": sampler_evaluation.models.german_credit(),
            "Stochastic_Volatility_MAMS": stochastic_volatility_mams_paper,
            "Item_Response": item_response(),
        },
```


In most cases, an inference gym model is used. For `Rosenbrock_36D()`, we provide our own product of 18 2D Rosenbrocks. For `stochastic_volatility_mams_paper`, we adapt the NumPyro Stochastic Volatility model, which differs from inference gym in number of dimensions and presents a harder inference problem (more steps required for convergence by a factor of 10).  For `banana_mams_paper`, our implementation aligns with inference gym, and is present to avoid [a apparent bug incurred when using 64 bit precision](https://github.com/tensorflow/probability/issues/1993). For `IllConditionedGaussian`, we again provide our own variant of a Gaussian, which is unrotated.

For the models marked with "MAMS", please disregard the ground truths of the first moments (e.g. `{"identity": SampleTransformation(ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf)}`), which are set to infinity and not used in the MAMS Paper.

Ground truth values for the expectation of $x^2$ are generated in the `sampler-evaluation` package, in `sampler_evaluation/models/data/estimate_expectations.py`, with the exception of `stochastic_volatility_mams_paper`, where a pre-existing ground truth was used.

## Table 2

These results were generated with `sampler_comparison/experiments/grid_search_comparison.py`. For `stochastic_volatility_mams_paper`, 32 chains were used.


Todos:

- include the Stoch vol ground truth (currently not in repo)
- fix the paths so they aren't specific to you!
- break experiment into separate files!s
