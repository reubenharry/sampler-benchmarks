# Benchmarking Samplers

The purpose of this package is to run **[Blackjax sampling algorithms](https://blackjax-devs.github.io/blackjax/)** on models from **[Inference Gym](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb)**, and to collect statistics measuring performance from the `sampler-evaluation` package.

This package also serves to demonstrate best practices for running samplers (tuning schemes, preconditioning, choice of integrator, etc.).

**It is currently under development**, which is to say that more samplers are forthcoming, and the API may change.

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

# Results

See [here](./results/) for the evaluation of each sampler on each model, which can be reproduced with `./results/run_benchmarks.py` and viewed in `./examples/demo.ipynb`

As the package is developed, the goal is to expand the set of models, samplers and statistics. **Anyone is welcome to contribute a new sampler, model or statistic!**

# Installation

Currently the package is not on PyPI, so you will need to clone the repository and install it locally.
