This repository contains two Python packages, `sampler-evaluation` and `sampler-comparison`. 

`sampler-evaluation` has code to take an array of samples of shape `[num_chains, num_samples, num_dimensions]` and return an estimate of the effective sample size (ESS).

`sampler-comparison` provides baseline versions of a number of samplers, and depends on `sampler-evaluation` to collect ESS estimates.

**This code is still a draft, not even alpha quality.** It is not yet ready for use by others, but is being developed in the open to solicit feedback and contributions.
