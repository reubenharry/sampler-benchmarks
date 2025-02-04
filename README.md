This repository contains two Python packages, `sampler-evaluation` and `sampler-comparison`. 

`sampler-evaluation` has code to take an array of samples of shape `[num_chains, num_samples, num_dimensions]` and return an estimate of the effective sample size (ESS).

`sampler-comparison` provides baseline versions of a number of samplers, and depends on `sampler-evaluation` to collect ESS estimates.
