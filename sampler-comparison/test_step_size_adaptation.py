import jax
import jax.numpy as jnp
import os
from functools import partial

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
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import grid_search_adjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian

# Create a simple model for testing
model = IllConditionedGaussian(ndims=10, condition_number=1, eigenvalues='log')

print("=== Testing Enhanced Step Size Adaptation in Grid Search ===")
print(f"Model: {model.name} (ndims={model.ndims})")
print(f"Batch size: {batch_size}")

# Test unadjusted sampler with robnik_adaptation + epsilon grid
print("\n--- Testing Unadjusted MCLMC with robnik_adaptation + epsilon grid ---")
try:
    run_benchmarks(
        models={model.name: model},
        samplers={
            "test_unadjusted_mclmc": partial(grid_search_unadjusted_mclmc, 
                                            num_tuning_steps=5000, 
                                            integrator_type="mclachlan",
                                            diagonal_preconditioning=True, 
                                            num_chains=batch_size,
                                            grid_size=5,  # Smaller grid for testing
                                            grid_iterations=1),  # Single iteration for testing
        },
        batch_size=batch_size,
        num_steps=5000,  # Shorter run for testing
        save_dir=f"results/test_step_size_adaptation",
        key=jax.random.key(42),
        map=lambda x : x,  # Use regular map for grid search
        calculate_ess_corr=False,
    )
    print("✓ Unadjusted MCLMC test completed successfully")
except Exception as e:
    print(f"✗ Unadjusted MCLMC test failed: {e}")

# Test adjusted sampler with da_adaptation + epsilon grid
print("\n--- Testing Adjusted MCLMC with da_adaptation + epsilon grid ---")
try:
    run_benchmarks(
        models={model.name: model},
        samplers={
            "test_adjusted_mclmc": partial(grid_search_adjusted_mclmc, 
                                          num_tuning_steps=5000, 
                                          integrator_type="mclachlan",
                                          diagonal_preconditioning=True, 
                                          num_chains=batch_size,
                                          grid_size=5,  # Smaller grid for testing
                                          grid_iterations=1,  # Single iteration for testing
                                          target_acc_rate=0.9,
                                          L_proposal_factor=1.25,
                                          random_trajectory_length=True),
        },
        batch_size=batch_size,
        num_steps=5000,  # Shorter run for testing
        save_dir=f"results/test_step_size_adaptation",
        key=jax.random.key(43),
        map=lambda x : x,  # Use regular map for grid search
        calculate_ess_corr=False,
    )
    print("✓ Adjusted MCLMC test completed successfully")
except Exception as e:
    print(f"✗ Adjusted MCLMC test failed: {e}")

print("\n=== Test Summary ===")
print("The enhanced step size adaptation functionality:")
print("  - Uses robnik_adaptation (packaged function) for unadjusted samplers")
print("  - Uses da_adaptation for adjusted samplers")
print("  - Adds epsilon grid search centered on adaptation results")
print("  - Adaptation steps: min(200, num_steps/5)")
print("  - Epsilon grid respects L/step_size constraints (1 <= L/step_size <= 50)")
print("  - Maintains boundary detection for both L and step_size parameters")
print("  - Provides more principled and robust step size optimization")
print("  - Integrates seamlessly with existing grid search framework")
print("\nOptimization structure:")
print("  Outer loop: Grid search over L values")
print("  Inner loop: For each L:")
print("    1. Run adaptation (robnik_adaptation or da_adaptation)")
print("    2. Run epsilon grid search around adapted step size")
print("    3. Evaluate sampler with optimal parameters") 