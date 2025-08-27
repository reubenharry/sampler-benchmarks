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

from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian

# Create a simple model for testing
model = IllConditionedGaussian(ndims=10, condition_number=1, eigenvalues='log')

# Example usage of the new grid search function
print("Running new grid search for unadjusted MCLMC...")
print("The function will automatically use model-specific preferences from model_info")

run_benchmarks(
    models={model.name: model},
    samplers={
        "grid_search_mclmc": grid_search_unadjusted_mclmc(
            num_chains=batch_size,
            integrator_type="mclachlan",
            grid_size=8,
            grid_iterations=2,
            # The function will automatically use model-specific preferences:
            # - statistic: from model_info['preferred_statistic']
            # - max_over_parameters: from model_info['max_over_parameters'] 
            # - grid_search_steps: from model_info['grid_search_steps']
        ),
    },
    batch_size=batch_size,
    num_steps=5000,  # Shorter for testing
    save_dir=f"results/Grid_Search_Test",
    key=jax.random.key(42),
    map=lambda x: x,
    calculate_ess_corr=False,
)

print("Grid search completed! Check the results directory for output.")
print("The function used model-specific preferences automatically.") 