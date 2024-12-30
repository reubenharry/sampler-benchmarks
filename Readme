# Benchmarking Samplers

The purpose of this package is to run **[Blackjax sampling algorithms](https://blackjax-devs.github.io/blackjax/)** on models from **[Inference Gym](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb)**, and to collect statistics measuring effective sample size.

**It is currently under development**

# Usage

```python
samples, metadata = samplers.nuts(return_samples=True)(
        model=gym.targets.Banana(),
        num_steps=1000,
        initial_position=jnp.array([1., 1.]),
        key=jax.random.PRNGKey(0))
```

`samples` is then an array of samples from the distribution:

![banana](./img/banana.png)

# Metrics

**Under construction**

# Features

- Expectations are (by default) calculated online, so that samples can be discarded as they are generated. This is useful for high dimensional long chains, where it is not feasible to store all samples in memory.

- Grid search: under construction

- Benchmark runs
