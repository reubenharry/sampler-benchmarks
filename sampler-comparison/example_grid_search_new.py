import jax
import jax.numpy as jnp
import os

# Set up JAX for multi-device execution
batch_size = 64
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers

from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc_new
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian

# Create a simple model for testing
model = IllConditionedGaussian(ndims=10, condition_number=1, eigenvalues='log')

# Example usage of the new grid search function
print("Running new grid search for unadjusted MCLMC...")

run_benchmarks(
    models={model.name: model},
    samplers={
        "grid_search_mclmc_new": grid_search_unadjusted_mclmc_new(
            num_chains=batch_size,
            integrator_type="mclachlan",
            initial_L=5.0,
            epsilon=0.1,
            grid_size=8,
            grid_iterations=2,
        ),
    },
    batch_size=batch_size,
    num_steps=5000,  # Shorter for testing
    save_dir=f"results/Grid_Search_New_Test",
    key=jax.random.key(42),
    map=lambda x: x,
    calculate_ess_corr=False,
)

print("Grid search completed! Check the results directory for output.") 