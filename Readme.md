# Benchmarking Samplers

The purpose of this package is to run **[Blackjax sampling algorithms](https://blackjax-devs.github.io/blackjax/)** on models from **[Inference Gym](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb)**, and to collect statistics measuring effective sample size.

This package also serves to demonstrate best practices for running samplers (tuning schemes, preconditioning, choice of integrator, etc.).

**It is currently under development**, which is to say that more models and samplers are forthcoming, and the API may change.

# Usage

## Running a sampler on a model

```python
samples, metadata = samplers['nuts'](return_samples=True)(
        model=gym.targets.Banana(),
        num_steps=1000,
        initial_position=jnp.ones(2),
        key=jax.random.PRNGKey(0))
```

`samples` is then an array of samples from the distribution:

![banana](./img/banana.png)

(See examples/demo.ipynb for the complete example with imports)

## Evaluating samplers

There is a range of ways to measure how efficient a sampler is, and the eventual goal is to provide a wide but standard set of such diagnostics.

For example, we can track the statistics $\frac{(E_{\mathit{sampler}}[x^2]-E[x^2])^2}{Var[x^2]}$ and $\frac{(E_{\mathit{sampler}}[x]-E[x])^2}{Var[x]}$, where $E_{\mathit{sampler}}$ is the empirical estimate of the expectation. We can then count how many steps of the kernel (and in particular for gradient based samplers, how many gradient calls), it takes for these statistics to drop below a threshold (by default $0.01$). In code:

```python
(
    stats,
    squared_errors,
) = sampler_grads_to_low_error(
    sampler=nuts(),
    model=Gaussian(ndims=10),
    num_steps=1000,
    batch_size=10,
    key=key,
    pvmap=jax.pmap
)
```

Then `stats['avg_over_parameters']['square']['grads_to_low_error']` is the number of steps it took for the squared error of $x^2$ (averaged across dimensions) to drop below the threshold, for a 10-dimensional Gaussian and the No-U-Turn Hamiltonian Monte Carlo (NUTS HMC) sampler.

Since not all inference gym models have a known expectation $E[x^2]$ or $E[x^4], blackjax-benchmarks calculates these numerically when unknown, using a long run of NUTS HMC. This code to generate these expectations is found in `./models/data/estimate_expectations.py`.

Meanwhile, see `./samplers/__init__.py` for a list of available samplers (everything here can simply be run out of the box - the tuning scheme and hyperparameters have all been chosen) and `./models/__init__.py` for a list of available models (with analytically known or )

<!-- Since gradient calls are the main computational expense of the sampler, and since $E[x^2]$ is a non-trivial statistic of a distribution, this metric is a good proxy for how long (in wallclock time) it takes a sampler to get good results on a given model.  -->

# Results

See [here](./results/) for the evaluation of each sampler on each model, which can be reproduced with `./results/run_benchmarks.py` and viewed in `./examples/demo.ipynb`

As the package is developed, the goal is to expand the set of models, samplers and statistics. **Anyone is welcome to contribute a new sampler, model or statistic!**

# Installation

Currently the package is not on PyPI, so you will need to clone the repository and install it locally.