import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import sys
import gc

sys.path.append("../sampler-comparison")
from sampler_comparison.util import map_integrator_type_to_integrator
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc_no_tuning,
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc_no_tuning,
)
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_make_L_step_size_adaptation,
)
from blackjax.adaptation.mclmc_adaptation import make_L_step_size_adaptation
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from sampler_comparison.samplers.general import sampler_grads_to_low_error
from sampler_comparison.samplers.general import (
    make_log_density_fn,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import (
    adjusted_hmc_no_tuning,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc_no_tuning
import blackjax.mcmc.metrics as metrics
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import unadjusted_hmc_no_tuning
from blackjax.adaptation.unadjusted_alba import unadjusted_alba
from typing import Callable

def clear_jax_cache():
    """Clear JAX compilation cache to prevent memory accumulation during grid search"""
    try:
        jax.clear_caches()
        print("    ✓ Cleared JAX compilation cache")
    except Exception as e:
        print(f"    ⚠ Warning: Could not clear JAX cache: {e}")
    
    # Force garbage collection
    gc.collect()

def grid_search_1D(
    objective_fn,
    coordinates,
    key,
    coordinate_name="coordinate",
):
    """
    General 1D grid search that minimizes an arbitrary function over a 1D array of coordinates.
    
    Args:
        objective_fn: Function that takes a coordinate and returns a scalar value to minimize
        coordinates: 1D array of coordinates to evaluate
        key: JAX random key
        coordinate_name: Name of the coordinate being optimized (e.g., "L", "epsilon")
    
    Returns:
        Tuple of (optimal_coordinate, optimal_value, all_values, optimal_index)
    """
    values = jnp.zeros(len(coordinates))
    
    print(f"    Evaluating {len(coordinates)} {coordinate_name} values:")
    for i, coord in enumerate(coordinates):
        eval_key = jax.random.fold_in(key, i)
        value = objective_fn(coord, eval_key)
        values = values.at[i].set(value)
        print(f"      {coordinate_name.capitalize()} {i+1}/{len(coordinates)}: {coord:.4f} -> {value:.4f}")
        
        # Clear cache every few evaluations to prevent memory accumulation
        if (i + 1) % 5 == 0:
            clear_jax_cache()
    
    optimal_idx = jnp.argmin(values)
    optimal_coordinate = coordinates[optimal_idx]
    optimal_value = values[optimal_idx]
    
    print(f"    Optimal {coordinate_name}: {optimal_coordinate:.4f} (index {optimal_idx}) with value: {optimal_value:.4f}")
    
    return optimal_coordinate, optimal_value, values, optimal_idx

def robnik_adaptation(
    algorithm,
    logdensity_fn: Callable,
    inverse_mass_matrix,
    initial_step_size: float = 1.0,
    desired_energy_var: float = 5e-4,
    initial_L: float = 1.0,
    integrator=blackjax.mcmc.integrators.velocity_verlet,
):
    """
    Robnik step size adaptation for unadjusted samplers.
    
    Args:
        algorithm: The algorithm to use (e.g., blackjax.mclmc, blackjax.langevin)
        logdensity_fn: Log density function
        inverse_mass_matrix: Inverse mass matrix
        initial_step_size: Initial step size
        desired_energy_var: Desired energy variance
        initial_L: Initial L value
        integrator: Integrator to use
    
    Returns:
        AdaptationAlgorithm with run method that returns (state, params, info)
    """
    from blackjax.adaptation.unadjusted_step_size import robnik_step_size_tuning
    
    # Initialize robnik step size tuning
    robnik_init, robnik_update, robnik_final = robnik_step_size_tuning(
        desired_energy_var=desired_energy_var
    )
    
    # Create kernel for the algorithm
    kernel = algorithm.build_kernel(integrator=integrator)
    
    def step(state, key):
        (adaptation_state, kernel_state), L = state
        
        # Run one step of the sampler
        new_kernel_state, info = kernel(
            rng_key=key,
            state=kernel_state,
            logdensity_fn=logdensity_fn,
            step_size=adaptation_state.step_size,
            L=L,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        
        # Update robnik state
        new_adaptation_state = robnik_update(adaptation_state, info)
        
        return ((new_adaptation_state, new_kernel_state), L), info
    
    def run(rng_key: jax.random.PRNGKey, position: jax.Array, num_steps: int = 1000):
        init_key, rng_key = jax.random.split(rng_key)
        
        # Initialize kernel state
        init_kernel_state = algorithm.init(
            position=position,
            logdensity_fn=logdensity_fn,
            random_generator_arg=init_key,
        )
        
        # Initialize robnik state
        init_adaptation_state = robnik_init(initial_step_size, position.shape[0])
        
        # Run adaptation
        keys = jax.random.split(rng_key, num_steps)
        init_state = (init_adaptation_state, init_kernel_state)
        ((final_adaptation_state, final_kernel_state), L), info = jax.lax.scan(
            step,
            (init_state, initial_L),
            keys,
        )
        
        # Get final step size
        optimal_step_size = robnik_final(final_adaptation_state)
        
        return (
            final_kernel_state,
            {
                "step_size": optimal_step_size,
                "inverse_mass_matrix": inverse_mass_matrix,
                "L": L,
            },
            info,
        )
    
    return blackjax.base.AdaptationAlgorithm(run)


def grid_search_L(
    model,
    num_gradient_calls,  # Changed from num_steps to num_gradient_calls
    num_chains,
    integrator_type,
    key,
    initial_L,
    initial_inverse_mass_matrix,
    initial_step_size,
    initial_state,
    sampler_fn,
    algorithm,
    integrator,
    statistic="square",
    max_over_parameters=True,
    grid_size=5,
    grid_iterations=2,
    is_adjusted_sampler=False,
    target_acc_rate=0.9,
    desired_energy_var=5e-4,
    L_proposal_factor=jnp.inf,
    random_trajectory_length=True,
):
    """
    Cleaner and more principled grid search for L parameter using any sampler function.
    Also optimizes step_size for each L value using proper adaptation algorithms.
    
    Args:
        model: The model to sample from
        num_gradient_calls: Target number of gradient calls (constant across integrators/samplers)
        num_chains: Number of chains to run
        integrator_type: Type of integrator to use
        key: JAX random key
        initial_L: Initial value of L parameter
        initial_inverse_mass_matrix: Initial inverse mass matrix
        initial_step_size: Initial step size to use (from ALBA warmup)
        initial_state: Initial state for the sampler
        sampler_fn: Function that creates a sampler (e.g., unadjusted_mclmc_no_tuning)
        algorithm: The algorithm to use for adaptation (e.g., blackjax.mclmc, blackjax.adjusted_mclmc_dynamic)
        integrator: The integrator to use for adaptation
        statistic: Which statistic to optimize ("square", "abs", etc.)
        max_over_parameters: Whether to use max_over_parameters (True) or avg_over_parameters (False)
        grid_size: Number of L values to try in each grid iteration
        grid_iterations: Number of grid search iterations
        is_adjusted_sampler: Whether this is an adjusted sampler (affects step size adaptation method)
        target_acc_rate: Target acceptance rate for adjusted samplers
        desired_energy_var: Desired energy variance for unadjusted samplers
        L_proposal_factor: Factor for L proposal in adjusted samplers
        random_trajectory_length: Whether to use random trajectory length for adjusted samplers
    
    Returns:
        Tuple of (optimal_L, optimal_step_size, optimal_value, all_values, optimal_index, tuning_outcome)
    """
    
    def calculate_num_steps(L, step_size):
        """
        Calculate the number of steps needed to achieve num_gradient_calls.
        
        Gradient calls per sample:
        - Unadjusted + Velocity Verlet: 1 gradient call per sample
        - Unadjusted + Mclachlan/Omelyan: 2 gradient calls per sample  
        - Adjusted + Velocity Verlet: L/step_size gradient calls per sample
        - Adjusted + Mclachlan/Omelyan: 2 × L/step_size gradient calls per sample
        """
        # Determine gradient calls per integration step based on integrator
        if integrator_type in ['mclachlan', 'omelyan']:
            grads_per_integration_step = 2
        else:  # velocity_verlet
            grads_per_integration_step = 1
        
        if is_adjusted_sampler:
            # For adjusted samplers: L/step_size integration steps per sample
            integration_steps_per_sample = L / step_size
            grads_per_sample = grads_per_integration_step * integration_steps_per_sample
        else:
            # For unadjusted samplers: 1 integration step per sample
            grads_per_sample = grads_per_integration_step
        
        # Calculate number of steps needed
        num_steps = int(num_gradient_calls / grads_per_sample)
        return max(1, num_steps)  # Ensure at least 1 step
    
    def objective_fn(L, eval_key):
        """Objective function to minimize: number of gradient evaluations per effective sample"""
        
        print(f"    Testing L = {L:.4f} (inner loop: optimizing step_size)...")
        
        # Calculate adaptation steps: minimum of 200 and 1/5 of num_steps
        # Use a reasonable step size estimate for adaptation steps calculation
        estimated_step_size = initial_step_size
        adaptation_num_steps = calculate_num_steps(L, estimated_step_size)
        adaptation_steps = min(200, adaptation_num_steps // 5)
        
        if is_adjusted_sampler:
            # For adjusted samplers, use da_adaptation
            from blackjax.adaptation.adjusted_alba import da_adaptation, make_random_trajectory_length_fn
            
            # Create integration steps function
            integration_steps_fn = make_random_trajectory_length_fn(random_trajectory_length)
            
            # Run da_adaptation to find optimal step size for this L
            da_warmup = da_adaptation(
                algorithm=algorithm,
                logdensity_fn=make_log_density_fn(model),
                integration_steps_fn=integration_steps_fn,
                inverse_mass_matrix=initial_inverse_mass_matrix,
                initial_step_size=initial_step_size,
                target_acceptance_rate=target_acc_rate,
                initial_L=L,
                integrator=integrator,
                L_proposal_factor=L_proposal_factor,
            )
            
            # Run adaptation
            adapted_state, adapted_params, _ = da_warmup.run(eval_key, initial_state.position, adaptation_steps)
            optimal_step_size = adapted_params["step_size"]
            
        else:
            # For unadjusted samplers, use robnik_adaptation
            # Run robnik adaptation to find optimal step size for this L
            robnik_warmup = robnik_adaptation(
                algorithm=algorithm,
                logdensity_fn=make_log_density_fn(model),
                inverse_mass_matrix=initial_inverse_mass_matrix,
                initial_step_size=initial_step_size,
                desired_energy_var=desired_energy_var,
                initial_L=L,
                integrator=integrator,
            )
            
            # Run adaptation
            adapted_state, adapted_params, _ = robnik_warmup.run(eval_key, initial_state.position, adaptation_steps)
            optimal_step_size = adapted_params["step_size"]
        
        # Now do a grid search around the optimal step size found by adaptation
        print(f"      Running epsilon grid search around {optimal_step_size:.6f}...")
        
        # Define step_size range centered on the adapted step size
        # We want 1 <= L/step_size <= 50, so step_size should be in [L/50, L/1]
        min_step_size = L / 50.0
        max_step_size = L / 1.0
        
        # Ensure we don't go below a reasonable minimum step size
        min_step_size = max(min_step_size, 0.001)
        
        # Create a grid centered on the adapted step size, but within the L/step_size constraints
        center_step_size = optimal_step_size
        center_step_size = jnp.clip(center_step_size, min_step_size, max_step_size)
        
        # Create a grid around the center, ensuring we stay within bounds
        grid_radius = min(center_step_size * 0.5, (max_step_size - min_step_size) / 2)
        grid_min = max(center_step_size - grid_radius, min_step_size)
        grid_max = min(center_step_size + grid_radius, max_step_size)
        
        step_size_range = jnp.linspace(grid_min, grid_max, 10)
        
        def step_size_objective_fn(step_size, step_size_eval_key):
            """Inner objective function for step_size optimization"""
            # Calculate the number of steps needed for this step_size to achieve num_gradient_calls
            num_steps = calculate_num_steps(L, step_size)
            jax.debug.print("num_steps: {x}", x=num_steps)
            
            sampler = sampler_fn(
                initial_state=initial_state,
                integrator_type=integrator_type,
                step_size=step_size,
                L=L,
                inverse_mass_matrix=initial_inverse_mass_matrix,
                return_samples=False,
            )
        
            (stats, sq_error) = sampler_grads_to_low_error(
                    model=model,
                    sampler=jax.pmap(
                lambda key, pos: sampler(
                    model=model, num_steps=num_steps, initial_position=pos, key=key
                    )
                ),
                    key=step_size_eval_key,
                    batch_size=num_chains,
                )

            param_type = "max_over_parameters" if max_over_parameters else "avg_over_parameters"
            return stats[param_type][statistic]["grads_to_low_error"]
        
        # Step size grid search with automatic expansion
        optimal_step_size, optimal_step_size_value, step_size_values, step_size_optimal_idx, step_size_tuning_outcome = step_size_grid_search_with_expansion(
            objective_fn=step_size_objective_fn,
            center_step_size=optimal_step_size,
            L=L,
            key=eval_key,
            grid_size=grid_size,
        )
        
        # Track step size boundary hits for this L value
        step_size_boundary_hits.append({
            'L': L,
            'tuning_outcome': step_size_tuning_outcome,
            'optimal_step_size': optimal_step_size,
            'optimal_value': optimal_step_size_value
        })
        
        print(f"    L = {L:.4f} complete: optimal step_size = {optimal_step_size:.6f}, value = {optimal_step_size_value:.4f}")
        
        return optimal_step_size_value, step_size_tuning_outcome
    
    def step_size_grid_search_with_expansion(objective_fn, center_step_size, L, key, grid_size=5):
        """
        Step size grid search with automatic boundary expansion.
        
        Args:
            objective_fn: Function that takes (step_size, key) and returns a scalar value to minimize
            center_step_size: Step size to center the search around
            L: Current L value (used for bounds calculation)
            key: JAX random key
            grid_size: Number of step sizes to evaluate in each grid search
        
        Returns:
            Tuple of (optimal_step_size, optimal_value, all_values, optimal_index, tuning_outcome)
        """
        # Define step_size range using multiplicative grid centered on center_step_size
        # Create grid with factors [0.5, 0.7, 1.0, 1.4, 2.0] for initial search
        factors = jnp.array([0.5, 0.7, 1.0, 1.4, 2.0])
        if grid_size != 5:
            # For different grid sizes, create symmetric multiplicative factors
            n_half = grid_size // 2
            if grid_size % 2 == 1:  # odd size
                factors = jnp.concatenate([
                    jnp.array([0.5, 0.7, 1.0]),
                    jnp.array([1.4, 2.0])
                ])
            else:  # even size
                factors = jnp.concatenate([
                    jnp.array([0.5, 0.7]),
                    jnp.array([1.4, 2.0])
                ])
        
        step_size_range = center_step_size * factors
        
        # Apply bounds: ensure 1 <= L/step_size <= 50
        min_step_size = L / 50.0
        max_step_size = L / 1.0
        min_step_size = max(min_step_size, 0.001)  # reasonable minimum
        
        # Clip to bounds
        step_size_range = jnp.clip(step_size_range, min_step_size, max_step_size)
        
        # Track expansion iteration (for boundary expansion only)
        expansion_iteration = 0
        max_expansions = 2  # Up to 4 total attempts (original + 3 expansions)
        tuning_outcome = 0  # 0 = success, 1 = boundary_failure_inf, 2 = boundary_failure
        
        while expansion_iteration <= max_expansions:
            print(f"      Step size grid search attempt {expansion_iteration + 1}/{max_expansions + 1}")
            if expansion_iteration > 0:
                print(f"      Range: [{step_size_range[0]:.6f}, {step_size_range[-1]:.6f}]")
            
            # Run grid search
            optimal_step_size, optimal_step_size_value, step_size_values, step_size_optimal_idx = grid_search_1D(
                objective_fn=objective_fn,
                coordinates=step_size_range,
                key=key,
                coordinate_name="step_size",
            )
            
            # Check if we hit a boundary
            hit_lower_boundary = step_size_optimal_idx == 0
            hit_upper_boundary = step_size_optimal_idx == len(step_size_range) - 1
            
            # If we didn't hit any boundary, we're done
            if not hit_lower_boundary and not hit_upper_boundary:
                print(f"      Found optimal step_size {optimal_step_size:.6f} within range")
                break
            
            # If we hit a boundary and the optimal value is inf, return the boundary value
            if jnp.isinf(optimal_step_size_value):
                print(f"      Optimal value is inf - returning boundary value without expansion")
                tuning_outcome = 1 # boundary_failure_inf
                break
            
            # If we've reached max expansions, continue with the best value found
            if expansion_iteration == max_expansions:
                boundary_type = "lower" if hit_lower_boundary else "upper"
                print(f"      WARNING: Step size grid search failed: still hitting {boundary_type} boundary after {max_expansions} expansions.")
                print(f"      Continuing with best value found: {optimal_step_size:.6f}")
                tuning_outcome = 2 # boundary_failure
                break
            
            # Expand the range for the next iteration using multiplicative factors
            expansion_factor = 2.0 ** (expansion_iteration + 1)  # 2, 4, 8
            
            if hit_lower_boundary:
                print(f"      WARNING: Optimal step_size {optimal_step_size:.6f} hit LOWER boundary {step_size_range[0]:.6f}")
                print(f"      Expanding step_size grid range downward by factor {expansion_factor}...")
                # Use larger multiplicative factors for expansion
                factors = jnp.array([0.5/expansion_factor, 0.7/expansion_factor, 1.0, 1.4, 2.0])
                if grid_size != 5:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.array([0.5/expansion_factor, 0.7/expansion_factor, 1.0]),
                            jnp.array([1.4, 2.0])
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.array([0.5/expansion_factor, 0.7/expansion_factor]),
                            jnp.array([1.4, 2.0])
                        ])
            else:  # hit_upper_boundary
                print(f"      WARNING: Optimal step_size {optimal_step_size:.6f} hit UPPER boundary {step_size_range[-1]:.6f}")
                print(f"      Expanding step_size grid range upward by factor {expansion_factor}...")
                # Use larger multiplicative factors for expansion
                factors = jnp.array([0.5, 0.7, 1.0, 1.4*expansion_factor, 2.0*expansion_factor])
                if grid_size != 5:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.array([0.5, 0.7, 1.0]),
                            jnp.array([1.4*expansion_factor, 2.0*expansion_factor])
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.array([0.5, 0.7]),
                            jnp.array([1.4*expansion_factor, 2.0*expansion_factor])
                        ])
            
            step_size_range = center_step_size * factors
            step_size_range = jnp.clip(step_size_range, min_step_size, max_step_size)
            expansion_iteration += 1
        
        return optimal_step_size, optimal_step_size_value, step_size_values, step_size_optimal_idx, tuning_outcome
    
    optimal_L = initial_L
    optimal_step_size = initial_step_size
    optimal_value = None
    all_values = None
    optimal_idx = None
    overall_tuning_outcome = 0  # 0 = success, 1 = boundary_failure_inf, 2 = boundary_failure
    step_size_boundary_hits = []  # Track step size boundary hits for each L value
    L_boundary_hits = []  # Track L boundary hits for each grid iteration
    
    param_type = "max_over_parameters" if max_over_parameters else "avg_over_parameters"
    sampler_type = "adjusted" if is_adjusted_sampler else "unadjusted"
    print(f"\n=== Grid Search for L Parameter (with {sampler_type} step_size optimization) ===")
    print(f"Model: {model.name} (ndims={model.ndims})")
    print(f"Initial L: {initial_L:.4f}")
    print(f"Initial step_size: {initial_step_size:.6f}")
    print(f"Grid size: {grid_size}, Grid iterations: {grid_iterations}")
    print(f"Number of chains: {num_chains}, Target gradient calls: {num_gradient_calls}")
    print(f"Integrator: {integrator_type}")
    print(f"Statistic: {statistic}, Parameter type: {param_type}")
    print(f"Step size optimization: {'da_adaptation + epsilon grid' if is_adjusted_sampler else 'robnik_adaptation + epsilon grid'}")
    
    for grid_iteration in range(grid_iterations):
        grid_key = jax.random.fold_in(key, grid_iteration + 2)
        
        if grid_iteration == 0:
            # First iteration: search around initial_L with multiplicative grid
            # Use wider range for high-dimensional problems: [0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 7.0, 10.0]
            factors = jnp.array([0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 7.0, 10.0])
            if grid_size != 8:
                # For different grid sizes, create symmetric multiplicative factors
                n_half = grid_size // 2
                if grid_size % 2 == 1:  # odd size
                    factors = jnp.concatenate([
                        jnp.linspace(0.1, 1.0, n_half + 1),
                        jnp.linspace(1.0, 10.0, n_half)[1:]
                    ])
                else:  # even size
                    factors = jnp.concatenate([
                        jnp.linspace(0.1, 1.0, n_half),
                        jnp.linspace(1.0, 10.0, n_half)
                    ])
            L_range = initial_L * factors
            print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Initial Search) ---")
            print(f"  L range: {L_range}")
        else:
            # Subsequent iterations: refine around the best value found
            if optimal_idx is not None:
                # Create a finer multiplicative grid around the optimal value
                # Use smaller factors for refinement: [0.8, 0.9, 1.0, 1.1, 1.2]
                factors = jnp.array([0.8, 0.9, 1.0, 1.1, 1.2])
                if grid_size != 5:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.linspace(0.8, 1.0, n_half + 1),
                            jnp.linspace(1.0, 1.2, n_half)[1:]
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.linspace(0.8, 1.0, n_half),
                            jnp.linspace(1.0, 1.2, n_half)
                        ])
                L_range = optimal_L * factors
                print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Refinement) ---")
                print(f"  Refining around previous optimal L: {optimal_L:.4f}")
                print(f"  L range: [{L_range[0]:.4f}, {L_range[-1]:.4f}]")
            else:
                # Fallback: search around the current optimal_L
                factors = jnp.array([0.8, 0.9, 1.0, 1.1, 1.2])
                if grid_size != 5:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.linspace(0.8, 1.0, n_half + 1),
                            jnp.linspace(1.0, 1.2, n_half)[1:]
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.linspace(0.8, 1.0, n_half),
                            jnp.linspace(1.0, 1.2, n_half)
                        ])
                L_range = optimal_L * factors
                print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Fallback Refinement) ---")
                print(f"  L range: [{L_range[0]:.4f}, {L_range[-1]:.4f}]")
        
        print(f"  Starting L grid search (outer loop)...")
        
        # L grid search with boundary expansion (repeat current iteration up to 4 times if boundary hit)
        expansion_iteration = 0
        max_expansions = 3  # Up to 4 total attempts (original + 3 expansions)
        current_L_range = L_range.copy()
        
        while expansion_iteration <= max_expansions:
            print(f"    L grid search attempt {expansion_iteration + 1}/{max_expansions + 1}")
            if expansion_iteration > 0:
                print(f"    Range: [{current_L_range[0]:.4f}, {current_L_range[-1]:.4f}]")
            
            optimal_L, optimal_value, all_values, optimal_idx = grid_search_1D(
                objective_fn=lambda L, key: objective_fn(L, key)[0],  # Extract just the value, not the tuning outcome
                coordinates=current_L_range,
                key=grid_key,
                coordinate_name="L",
            )
            
            # Check if we hit a boundary
            hit_lower_boundary = optimal_idx == 0
            hit_upper_boundary = optimal_idx == len(current_L_range) - 1
            
            # Track boundary hits
            L_boundary_hits.append({
                'iteration': expansion_iteration + 1,
                'hit_lower': hit_lower_boundary,
                'hit_upper': hit_upper_boundary,
                'optimal_L': optimal_L,
                'range': (current_L_range[0], current_L_range[-1])
            })
            
            # If we didn't hit any boundary, we're done with this grid iteration
            if not hit_lower_boundary and not hit_upper_boundary:
                print(f"    Found optimal L {optimal_L:.4f} within range")
                break
            
            # If we hit a boundary and the optimal value is inf, return the boundary value
            if jnp.isinf(optimal_value):
                print(f"    Optimal value is inf - returning boundary value without expansion")
                overall_tuning_outcome = 1 # boundary_failure_inf
                break
            
            # If we've reached max expansions, continue with the best value found
            if expansion_iteration == max_expansions:
                boundary_type = "lower" if hit_lower_boundary else "upper"
                print(f"    WARNING: L grid search failed: still hitting {boundary_type} boundary after {max_expansions} expansions.")
                print(f"    Continuing with best value found: {optimal_L:.4f}")
                overall_tuning_outcome = 2 # boundary_failure
                break
            
            # Expand the range for the next attempt using multiplicative factors
            expansion_factor = 2.0 ** (expansion_iteration + 1)  # 2, 4, 8
            
            if hit_lower_boundary:
                print(f"    WARNING: Optimal L {optimal_L:.4f} hit LOWER boundary {current_L_range[0]:.4f}")
                print(f"    Expanding L grid range downward by factor {expansion_factor}...")
                # Use reasonable multiplicative factors for expansion
                factors = jnp.array([0.1/expansion_factor, 0.3/expansion_factor, 0.7, 1.0, 1.5, 3.0, 7.0, 10.0])
                if grid_size != 8:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.array([0.1/expansion_factor, 0.3/expansion_factor, 0.7, 1.0]),
                            jnp.array([1.5, 3.0, 7.0, 10.0])
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.array([0.1/expansion_factor, 0.3/expansion_factor]),
                            jnp.array([1.5, 3.0, 7.0, 10.0])
                        ])
            else:  # hit_upper_boundary
                print(f"    WARNING: Optimal L {optimal_L:.4f} hit UPPER boundary {current_L_range[-1]:.4f}")
                print(f"    Expanding L grid range upward by factor {expansion_factor}...")
                # Use reasonable multiplicative factors for expansion
                factors = jnp.array([0.1, 0.3, 0.7, 1.0, 1.5*expansion_factor, 3.0*expansion_factor, 7.0*expansion_factor, 10.0*expansion_factor])
                if grid_size != 8:
                    n_half = grid_size // 2
                    if grid_size % 2 == 1:  # odd size
                        factors = jnp.concatenate([
                            jnp.array([0.1, 0.3, 0.7, 1.0]),
                            jnp.array([1.5*expansion_factor, 3.0*expansion_factor, 7.0*expansion_factor, 10.0*expansion_factor])
                        ])
                    else:  # even size
                        factors = jnp.concatenate([
                            jnp.array([0.1, 0.3]),
                            jnp.array([1.5*expansion_factor, 3.0*expansion_factor, 7.0*expansion_factor, 10.0*expansion_factor])
                        ])
            
            current_L_range = optimal_L * factors
            expansion_iteration += 1
        
        print(f"  L grid search complete!")
        
        # Clear cache after each grid iteration
        print(f"  Clearing JAX cache after grid iteration {grid_iteration + 1}...")
        clear_jax_cache()
    
    print(f"\n=== Grid Search Complete ===")
    print(f"Final optimal L: {optimal_L:.4f}")
    print(f"Final optimal value: {optimal_value:.4f}")
    
    # Convert tuning outcome code to string for display
    tuning_outcome_str = {0: "success", 1: "boundary_failure_inf", 2: "boundary_failure"}[overall_tuning_outcome]
    print(f"Overall tuning outcome: {tuning_outcome_str}")
    
    # Report L boundary hits
    print(f"\n=== L Boundary Analysis ===")
    if L_boundary_hits:
        final_boundary_hit = L_boundary_hits[-1]
        if final_boundary_hit['hit_lower'] or final_boundary_hit['hit_upper']:
            boundary_type = "LOWER" if final_boundary_hit['hit_lower'] else "UPPER"
            print(f"⚠️  L grid search hit {boundary_type} boundary in final iteration")
            print(f"   Final L range: [{final_boundary_hit['range'][0]:.4f}, {final_boundary_hit['range'][1]:.4f}]")
            print(f"   Optimal L: {final_boundary_hit['optimal_L']:.4f}")
        else:
            print(f"✓ L grid search completed without hitting boundaries")
        
        # Count total boundary hits
        total_boundary_hits = sum(1 for hit in L_boundary_hits if hit['hit_lower'] or hit['hit_upper'])
        if total_boundary_hits > 0:
            print(f"   Total L boundary hits across all iterations: {total_boundary_hits}")
    else:
        print(f"✓ No L boundary hits recorded")
    
    # Report step size boundary hits
    print(f"\n=== Step Size Boundary Analysis ===")
    if step_size_boundary_hits:
        boundary_hit_count = sum(1 for hit in step_size_boundary_hits if hit['tuning_outcome'] > 0)
        inf_count = sum(1 for hit in step_size_boundary_hits if hit['tuning_outcome'] == 1)
        failure_count = sum(1 for hit in step_size_boundary_hits if hit['tuning_outcome'] == 2)
        
        if boundary_hit_count > 0:
            print(f"⚠️  Step size grid search hit boundaries for {boundary_hit_count}/{len(step_size_boundary_hits)} L values")
            if inf_count > 0:
                print(f"   - {inf_count} L values had infinite optimal values")
            if failure_count > 0:
                print(f"   - {failure_count} L values failed after max expansions")
            
            # Show details for problematic L values
            print(f"   Problematic L values:")
            for hit in step_size_boundary_hits:
                if hit['tuning_outcome'] > 0:
                    outcome_str = {1: "inf", 2: "boundary_failure"}[hit['tuning_outcome']]
                    print(f"     L={hit['L']:.4f}: {outcome_str}")
        else:
            print(f"✓ Step size grid search completed without hitting boundaries for any L value")
    else:
        print(f"✓ No step size boundary hits recorded")
    
    print(f"\nNote: Optimal step_size was found for each L value using {sampler_type} adaptation + epsilon grid")
    print(f"\nOptimization structure:")
    print(f"  Outer loop: Grid search over L values ({grid_iterations} iterations)")
    print(f"  Inner loop: For each L, {sampler_type} step_size adaptation + epsilon grid search")
    print(f"  Objective: Minimize gradient evaluations to low error")
    
    return optimal_L, optimal_step_size, optimal_value, all_values, optimal_idx, overall_tuning_outcome
