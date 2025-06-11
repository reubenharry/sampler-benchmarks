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

Since not all inference gym models have a known expectation $E[x^2]$ or $E[x^4], sampler-benchmarks calculates these numerically when unknown, using a long run of NUTS HMC. This code to generate these expectations is found in `./models/data/estimate_expectations.py`.

Meanwhile, see `./samplers/__init__.py` for a list of available samplers (everything here can simply be run out of the box - the tuning scheme and hyperparameters have all been chosen) and `./models/__init__.py` for a list of available models (with analytically known or )

<!-- Since gradient calls are the main computational expense of the sampler, and since $E[x^2]$ is a non-trivial statistic of a distribution, this metric is a good proxy for how long (in wallclock time) it takes a sampler to get good results on a given model.  -->
