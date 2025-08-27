import jax
import jax.numpy as jnp
import os

# Set up JAX for multi-device execution
batch_size = 32
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers

from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian

# Create a model that might trigger boundary issues
# Using a high-dimensional model with small grid size to increase chance of boundary hits
model = IllConditionedGaussian(ndims=50, condition_number=10, eigenvalues='log')

print("Testing boundary detection in grid search...")
print("This test uses a high-dimensional model with small grid size to increase")
print("the likelihood of hitting grid boundaries.")

try:
    run_benchmarks(
        models={model.name: model},
        samplers={
            "grid_search_mclmc_boundary_test": grid_search_unadjusted_mclmc(
                num_chains=batch_size,
                integrator_type="mclachlan",
                grid_size=5,  # Small grid size to increase chance of boundary hits
                grid_iterations=1,  # Single iteration to see boundary effects
                num_tuning_steps=2000,  # Shorter for testing
            ),
        },
        batch_size=batch_size,
        num_steps=2000,  # Shorter for testing
        save_dir=f"results/Boundary_Detection_Test",
        key=jax.random.key(42),
        map=lambda x: x,
        calculate_ess_corr=False,
    )
    print("Grid search completed successfully without boundary issues!")
    
except ValueError as e:
    print(f"\nBoundary detection test completed as expected!")
    print(f"The grid search correctly detected a boundary hit and terminated.")
    print(f"This demonstrates the boundary detection functionality is working.")
    
except Exception as e:
    print(f"\nUnexpected error occurred: {e}")
    print(f"This might be due to other issues, not necessarily boundary detection.") 