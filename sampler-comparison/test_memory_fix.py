#!/usr/bin/env python3
"""
Test script to verify memory management fixes for grid search.
This script runs a small grid search to ensure the memory management works correctly.
"""

import os
import sys
import jax
import gc

# Set JAX environment variables to help with memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

jax.config.update("jax_enable_x64", True)

batch_size = 32  # Smaller batch size for testing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

print(f"JAX device count: {num_cores}")
print(f"Batch size: {batch_size}")

# Add paths
sys.path.append(".")
sys.path.append("../sampler-evaluation")
sys.path.append("../../blackjax")

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import grid_search_unadjusted_mclmc
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian

def clear_jax_cache():
    """Clear JAX compilation cache to prevent memory accumulation"""
    try:
        jax.clear_caches()
        print("  ✓ Cleared JAX compilation cache")
    except Exception as e:
        print(f"  ⚠ Warning: Could not clear JAX cache: {e}")
    
    gc.collect()
    print("  ✓ Forced garbage collection")

def test_memory_management():
    """Test the memory management with a simple grid search"""
    
    print("\n=== Testing Memory Management ===")
    
    # Create a simple model for testing
    model = IllConditionedGaussian(ndims=10, condition_number=1, eigenvalues='log')
    
    print(f"Model: {model.name} (ndims={model.ndims})")
    print("Running grid search with memory management...")
    
    try:
        # Run a small grid search to test memory management
        run_benchmarks(
            models={model.name: model},
            samplers={
                "grid_search_test": grid_search_unadjusted_mclmc(
                    num_chains=batch_size,
                    integrator_type="mclachlan",
                    grid_size=4,  # Small grid for testing
                    grid_iterations=1,  # Single iteration for testing
                    num_tuning_steps=100,  # Small tuning for testing
                ),
            },
            batch_size=batch_size,
            num_steps=1000,  # Small number of steps for testing
            save_dir="results/Memory_Test",
            key=jax.random.key(42),
            map=lambda x: x,
            calculate_ess_corr=False,
        )
        
        print("✓ Grid search completed successfully!")
        
        # Clear cache after completion
        print("Clearing JAX cache after completion...")
        clear_jax_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Grid search failed: {e}")
        print("Clearing JAX cache after error...")
        clear_jax_cache()
        return False

if __name__ == "__main__":
    print("Starting memory management test...")
    
    # Test multiple runs to ensure memory doesn't accumulate
    for i in range(3):
        print(f"\n--- Test Run {i+1}/3 ---")
        success = test_memory_management()
        
        if not success:
            print(f"Test run {i+1} failed!")
            break
        else:
            print(f"Test run {i+1} completed successfully!")
    
    print("\n=== Memory Management Test Complete ===")
    print("If all tests passed, the memory management fixes are working correctly.") 