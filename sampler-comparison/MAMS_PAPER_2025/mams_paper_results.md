# [MAMS Paper]() Result generation

This document outlines the steps to generate the results for the MAMS paper, which was performed using the `sampler-comparison` package on the branch `research` of the `sampler-benchmarks` repository in which `sampler-comparison` is contained.

## Table 1

These results were generated with the scripts in the `table1` folder.

### Models

The model set is:

```python
models={
        banana_mams_paper,
        IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
        Rosenbrock_36D(),
        neals_funnel_mams_paper,
        brownian_motion(),
        sampler_evaluation.models.german_credit(),
        stochastic_volatility_mams_paper,
        item_response(),
        },
```

all of which are defined in the `sampler-evaluation` package (also in this repository), under `sampler_evaluation/models`.


In most cases, an inference gym model is used. For `Rosenbrock_36D()`, we provide our own product of 18 2D Rosenbrocks. For `stochastic_volatility_mams_paper`, we adapt the NumPyro Stochastic Volatility model, which differs from inference gym in number of dimensions and presents a harder inference problem (more steps required for convergence by a factor of 10).  For `banana_mams_paper`, our implementation aligns with inference gym, and is present to avoid [a apparent bug incurred when using 64 bit precision](https://github.com/tensorflow/probability/issues/1993). For `IllConditionedGaussian`, we again provide our own variant of a Gaussian, which is unrotated. For Neal's Funnel, we use our own implementation, which should be similar or the same to the inference gym version.

For the models marked with "MAMS", please disregard the ground truths of the first moments (e.g. `{"identity": SampleTransformation(ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf)}`), which are set to infinity and not used in the MAMS Paper.

Ground truth values for the expectation of $x^2$ are generated in the `sampler-evaluation` package, in `sampler_evaluation/models/data/estimate_expectations.py`, with the exception of `stochastic_volatility_mams_paper`, where a pre-existing ground truth was used, and models where the analytic form of the expectation is known.

### Samplers

```python
samplers={

            "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=5000),
            "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            "nuts": lambda: nuts(num_tuning_steps=5000),
        }
```

This set of samplers was used for the results, with certain variations (number of tuning steps, target acceptance rate) which can be found in the scripts in the `table1` folder.


## Table 2

These results were generated with `sampler_comparison/experiments/grid_search_comparison.py`. For `stochastic_volatility_mams_paper`, 32 chains were used.
