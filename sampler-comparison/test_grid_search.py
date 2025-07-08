#!/usr/bin/env python3
"""
Simple test script to verify grid search functionality works with unadjusted_mclmc.
"""

import jax
import jax.numpy as jnp
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.models import Gaussian

def test_grid_search():
    """Test that grid search works with unadjusted_mclmc."""
    print("=== Testing Grid Search with unadjusted_mclmc ===")
    
    # Set up model
    model = Gaussian(ndims=5)  # Small model for quick testing
    num_steps = 500  # Fewer steps for quick testing
    
    # Test ALBA (default behavior)
    print("Testing ALBA adaptation...")
    alba_sampler = unadjusted_mclmc(
        integrator_type="mclachlan",
        num_tuning_steps=1000,  # Short tuning for testing
        grid_search=False,  # Use ALBA
    )
    
    initial_position = jax.random.normal(jax.random.key(1), (model.ndims,))
    alba_expectations, alba_metadata = alba_sampler(
        model=model,
        num_steps=num_steps,
        initial_position=initial_position,
        key=jax.random.key(2),
    )
    
    print(f"ALBA - L: {alba_metadata['L']}, step_size: {alba_metadata['step_size']}")
    print(f"ALBA - tuning grads: {alba_metadata['num_tuning_grads']}")
    
    # Test Grid Search
    print("\nTesting Grid Search...")
    grid_sampler = unadjusted_mclmc(
        integrator_type="mclachlan",
        num_tuning_steps=1000,  # Not used for grid search
        grid_search=True,  # Use grid search
        grid_size=5,  # Small grid for quick testing
        num_chains=3,  # Few chains for quick testing
    )
    
    initial_position = jax.random.normal(jax.random.key(3), (model.ndims,))
    grid_expectations, grid_metadata = grid_sampler(
        model=model,
        num_steps=num_steps,
        initial_position=initial_position,
        key=jax.random.key(4),
    )
    
    print(f"Grid Search - L: {grid_metadata['L']}, step_size: {grid_metadata['step_size']}")
    print(f"Grid Search - tuning grads: {grid_metadata['num_tuning_grads']}")
    
    print("\n=== Test completed successfully! ===")
    return alba_metadata, grid_metadata

if __name__ == "__main__":
    try:
        alba_metadata, grid_metadata = test_grid_search()
        print("✅ Grid search test passed!")
    except Exception as e:
        print(f"❌ Grid search test failed: {e}")
        import traceback
        traceback.print_exc() 