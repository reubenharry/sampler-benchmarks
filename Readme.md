# Benchmarking Samplers

The purpose of this package is to run **[Blackjax sampling algorithms](https://blackjax-devs.github.io/blackjax/)** on models from **[Inference Gym](https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb)**, and to collect statistics measuring effective sample size.

**It is currently under development**

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

## Measuring Sampler Efficiency

This package provides `evaluate_sampler`, which can be used as follows:


```python
for i, (sampler, model) in enumerate(itertools.product(samplers, models)):

    err_t_mean_max, grads_to_low_max, err_t_mean_avg, grads_to_low_avg, expectation = evaluate_sampler(sampler=samplers[sampler](),model=models[model], num_steps=10000, batch_size=2, key=key)
    
    # Append the results to the list
    results.append({
        'Sampler': sampler,
        'Model': model,
        'Grad evaluations to low error (avg)': grads_to_low_avg
    })

# Create the DataFrame
df = pd.DataFrame(results)
```

(See examples/demo.py for the complete example, with imports)

Here, the statistic of interest is how many gradient calls it takes (of the log density of the model) to permanently decrease bias below $0.01$, where bias means $(E_{\mathit{sampler}}[x^2]-E[x^2])^2/Var[x^2]$, and $E_{\mathit{sampler}}$ is the empirical estimate of the expectation.

Since not all inference gym models have a known expectation $E[x^2]$, blackjax-benchmarks re-exports a subset which do (src/models).

<!-- Since gradient calls are the main computational expense of the sampler, and since $E[x^2]$ is a non-trivial statistic of a distribution, this metric is a good proxy for how long (in wallclock time) it takes a sampler to get good results on a given model.  -->

# Features

- Expectations are (by default) calculated online, so that samples can be discarded as they are generated. This is useful for high dimensional long chains, where it is not feasible to store all samples in memory.

- Grid search: under construction

- Benchmark runs
