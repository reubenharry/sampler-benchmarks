import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import sys

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
    
    optimal_idx = jnp.argmin(values)
    optimal_coordinate = coordinates[optimal_idx]
    optimal_value = values[optimal_idx]
    
    print(f"    Optimal {coordinate_name}: {optimal_coordinate:.4f} (index {optimal_idx}) with value: {optimal_value:.4f}")
    
    return optimal_coordinate, optimal_value, values, optimal_idx

def grid_search_L_new(
    model,
    num_steps,
    num_chains,
    integrator_type,
    key,
    initial_L,
    initial_inverse_mass_matrix,
    initial_step_size,
    initial_state,
    sampler_fn,
    statistic="square",
    max_over_parameters=True,
    grid_size=10,
    grid_iterations=2,
):
    """
    Cleaner and more principled grid search for L parameter using any sampler function.
    Also optimizes step_size for each L value.
    
    Args:
        model: The model to sample from
        num_steps: Number of sampling steps
        num_chains: Number of chains to run
        integrator_type: Type of integrator to use
        key: JAX random key
        initial_L: Initial value of L parameter
        initial_inverse_mass_matrix: Initial inverse mass matrix
        initial_step_size: Initial step size to use (from ALBA warmup)
        initial_state: Initial state for the sampler
        sampler_fn: Function that creates a sampler (e.g., unadjusted_mclmc_no_tuning)
        statistic: Which statistic to optimize ("square", "abs", etc.)
        max_over_parameters: Whether to use max_over_parameters (True) or avg_over_parameters (False)
        grid_size: Number of L values to try in each grid iteration
        grid_iterations: Number of grid search iterations
    
    Returns:
        Tuple of (optimal_L, optimal_step_size, optimal_value, all_values, optimal_index)
    """
    
    def objective_fn(L, eval_key):
        """Objective function to minimize: number of gradient evaluations per effective sample"""
        
        print(f"    Testing L = {L:.4f} (inner loop: optimizing step_size)...")
        
        # Define step_size range based on L/step_size ratio constraints
        # We want 1 <= L/step_size <= 50
        # So step_size should be in [L/50, L/1]
        min_step_size = L / 50.0
        max_step_size = L / 1.0
        
        # Ensure we don't go below a reasonable minimum step size
        min_step_size = max(min_step_size, 0.001)
        
        step_size_range = jnp.linspace(min_step_size, max_step_size, 10)
        
        def step_size_objective_fn(step_size, step_size_eval_key):
            """Inner objective function for step_size optimization"""
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
        
        # Find optimal step_size for this L value
        optimal_step_size, optimal_step_size_value, _, _ = grid_search_1D(
            objective_fn=step_size_objective_fn,
            coordinates=step_size_range,
            key=eval_key,
            coordinate_name="step_size",
        )
        
        print(f"    L = {L:.4f} complete: optimal step_size = {optimal_step_size:.6f}, value = {optimal_step_size_value:.4f}")
        
        return optimal_step_size_value
    
    optimal_L = initial_L
    optimal_step_size = initial_step_size
    optimal_value = None
    all_values = None
    optimal_idx = None
    
    param_type = "max_over_parameters" if max_over_parameters else "avg_over_parameters"
    print(f"\n=== Grid Search for L Parameter (with step_size optimization) ===")
    print(f"Model: {model.name} (ndims={model.ndims})")
    print(f"Initial L: {initial_L:.4f}")
    print(f"Initial step_size: {initial_step_size:.6f}")
    print(f"Grid size: {grid_size}, Grid iterations: {grid_iterations}")
    print(f"Number of chains: {num_chains}, Number of steps: {num_steps}")
    print(f"Integrator: {integrator_type}")
    print(f"Statistic: {statistic}, Parameter type: {param_type}")
    print(f"Step size range: L/50 to L/1 (ensuring 1 <= L/step_size <= 50)")
    
    for grid_iteration in range(grid_iterations):
        grid_key = jax.random.fold_in(key, grid_iteration + 2)
        
        if grid_iteration == 0:
            # First iteration: search around initial_L
            L_range = optimal_L * jnp.linspace(0.5, 2.0, grid_size)
            print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Initial Search) ---")
            print(f"  L range: [{L_range[0]:.4f}, {L_range[-1]:.4f}]")
        else:
            # Subsequent iterations: refine around the best value found
            if optimal_idx is not None:
                # Create a finer grid around the optimal value
                L_min = L_range[max(optimal_idx - 1, 0)]
                L_max = L_range[min(optimal_idx + 1, len(L_range) - 1)]
                L_range = jnp.linspace(L_min, L_max, grid_size)
                print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Refinement) ---")
                print(f"  Refining around previous optimal L: {optimal_L:.4f}")
                print(f"  L range: [{L_range[0]:.4f}, {L_range[-1]:.4f}]")
            else:
                # Fallback: search around the current optimal_L
                L_range = optimal_L * jnp.linspace(0.8, 1.2, grid_size)
                print(f"\n--- Grid Iteration {grid_iteration + 1}/{grid_iterations} (Fallback Refinement) ---")
                print(f"  L range: [{L_range[0]:.4f}, {L_range[-1]:.4f}]")
        
        print(f"  Starting L grid search (outer loop)...")
        optimal_L, optimal_value, all_values, optimal_idx = grid_search_1D(
            objective_fn=objective_fn,
            coordinates=L_range,
            key=grid_key,
            coordinate_name="L",
        )
        print(f"  L grid search complete!")
    
    print(f"\n=== Grid Search Complete ===")
    print(f"Final optimal L: {optimal_L:.4f}")
    print(f"Final optimal value: {optimal_value:.4f}")
    print(f"Note: Optimal step_size was found for each L value during optimization")
    print(f"\nOptimization structure:")
    print(f"  Outer loop: Grid search over L values ({grid_iterations} iterations)")
    print(f"  Inner loop: For each L, grid search over step_size values (L/50 to L/1)")
    print(f"  Objective: Minimize gradient evaluations to low error")
    
    return optimal_L, optimal_step_size, optimal_value, all_values, optimal_idx

def grid_search_step_size(state, params, num_steps, da_key_per_iter):
    return params


def grid_search_L(
    model,
    num_steps,
    num_chains,
    integrator_type,
    key,
    grid_size,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    sampler_type='adjusted_mclmc',
    euclidean=False,
    L_proposal_factor=jnp.inf,
    target_expectation='square',
    desired_energy_var=5e-4,
    diagonal_preconditioning=True, 
    acc_rate=0.9,
    initial_state=None,
    initial_inverse_mass_matrix=None,
):
    logdensity_fn = make_log_density_fn(model)

    da_key, bench_key, init_pos_key, init_key, nuts_key = jax.random.split(key, 5)

    # Use provided initial state or create new one
    if initial_state is not None:
        blackjax_state_after_tuning = initial_state
        if initial_inverse_mass_matrix is not None:
            inverse_mass_matrix = initial_inverse_mass_matrix
        else:
            inverse_mass_matrix = jnp.ones((model.ndims,))
    else:
        # Original NUTS-based initialization
        initial_position = jax.random.normal(
            shape=(model.ndims,),
            key=init_pos_key,
        )

        alg = blackjax.nuts

        if acc_rate is None:
            nuts_acc_rate = 0.8
        else:
            nuts_acc_rate = acc_rate

        warmup = blackjax.window_adaptation(
            alg,
            logdensity_fn,
            target_acceptance_rate=nuts_acc_rate,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        )

        (blackjax_state_after_tuning, nuts_params), inf = warmup.run(
            nuts_key, initial_position, num_tuning_steps
        )
        blackjax_state_after_tuning = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=blackjax_state_after_tuning.position,
            logdensity_fn=logdensity_fn,
            random_generator_arg=init_key,
        )
        # jax.debug.print("blackjax_state_after_tuning\n\n {x}", x=blackjax_state_after_tuning.position.shape)

        if diagonal_preconditioning:
            inverse_mass_matrix = nuts_params["inverse_mass_matrix"]
        else:
            inverse_mass_matrix = jnp.ones((model.ndims,))

    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps)
    ).astype(jnp.int32)

    if sampler_type=='adjusted_mclmc':
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            L_proposal_factor=jnp.inf,
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
        )

    elif sampler_type=='unadjusted_mclmc':
        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.mcmc.mclmc.build_kernel(
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            L=L,
        )

    elif sampler_type=='adjusted_hmc':
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.dynamic_malt.build_kernel(
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        L_proposal_factor=L_proposal_factor,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        )

    elif sampler_type=='unadjusted_lmc':
        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.langevin.build_kernel(
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
    
    elif sampler_type=='unadjusted_hmc':
        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.uhmc.build_kernel(
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
    else:
        raise Exception('sampler not recognized')

    z = 1 if euclidean else jnp.sqrt(model.ndims)
    state = blackjax_state_after_tuning

    Num_Grads = jnp.zeros(grid_size)
    Num_Grads_AVG = jnp.zeros(grid_size)
    Num_Grads_COV = jnp.zeros(grid_size)
    STEP_SIZE = jnp.zeros(grid_size)

    if acc_rate is None:
        acc_rate = 0.9

    for grid_iteration in range(grid_iterations):
        da_key_per_iter = jax.random.fold_in(da_key, grid_iteration)
        bench_key_per_iter = jax.random.fold_in(bench_key, grid_iteration)

        if grid_iteration == 0:
            Lgrid = jnp.linspace(z / 3, z * 3, grid_size)
        else:
            Lgrid = jnp.linspace(Lgrid[max(iopt - 1, 0)], Lgrid[min(iopt + 1, Lgrid.shape[0]-1)], grid_size)

        for i in range(len(Lgrid)):
            da_key_per_iter = jax.random.fold_in(da_key_per_iter, i)
            bench_key_per_iter = jax.random.fold_in(bench_key_per_iter, i)

            params = MCLMCAdaptationState(
                L=Lgrid[i],
                step_size=Lgrid[i] / 5,
                inverse_mass_matrix=inverse_mass_matrix,
            )

            if sampler_type=='adjusted_mclmc':
                (blackjax_state_after_tuning, params, _, _) = (
                    adjusted_mclmc_make_L_step_size_adaptation(
                        kernel=kernel,
                        dim=model.ndims,
                        frac_tune1=0.1,
                        frac_tune2=0.0,
                        target=acc_rate,
                        diagonal_preconditioning=False,
                        fix_L_first_da=True,
                    )(state, params, num_steps, da_key_per_iter)
                )

                sampler = adjusted_mclmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        L_proposal_factor=jnp.inf,
                        random_trajectory_length=True,
                    )

            elif sampler_type=='unadjusted_mclmc':
                mclmc_state = blackjax.mcmc.mclmc.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                )

                (blackjax_state_after_tuning, params) = (
                        make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            diagonal_preconditioning=False,
                        )(mclmc_state, params, num_steps, da_key_per_iter)
                    )

                sampler = unadjusted_mclmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                    )
                
            elif sampler_type=='adjusted_hmc':
                (blackjax_state_after_tuning, params, _, _) = (
                    adjusted_mclmc_make_L_step_size_adaptation(
                        kernel=kernel,
                        dim=model.ndims,
                        frac_tune1=0.1,
                        frac_tune2=0.0,
                        target=acc_rate,
                        diagonal_preconditioning=False,
                        fix_L_first_da=True,
                    )(state, params, num_steps, da_key_per_iter)
                )

                sampler = adjusted_hmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        random_trajectory_length=True,
                    )
                
            elif sampler_type=='unadjusted_lmc':
                lmc_state = blackjax.mcmc.underdamped_langevin.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                )

                (blackjax_state_after_tuning, params) = make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            diagonal_preconditioning=False,
                            euclidean=True,
                            desired_energy_var=desired_energy_var,
                        )(lmc_state, params, num_steps, da_key_per_iter)
                
                sampler = unadjusted_lmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                    )

            elif sampler_type=='unadjusted_hmc':
                hmc_state = blackjax.mcmc.uhmc.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                )

                (blackjax_state_after_tuning, params) = make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            diagonal_preconditioning=False,
                            euclidean=True,
                            desired_energy_var=desired_energy_var,
                        )(hmc_state, params, num_steps, da_key_per_iter)
                
                sampler = unadjusted_hmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                    )
                
            

            (stats, sq_error) = sampler_grads_to_low_error(
                model=model,
                sampler=jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps, initial_position=pos, key=key
                )
            ),
                key=bench_key_per_iter,
                batch_size=num_chains,
            )

            Num_Grads = Num_Grads.at[i].set(
                stats["max_over_parameters"][target_expectation]["grads_to_low_error"]
            )
            Num_Grads_AVG = Num_Grads_AVG.at[i].set(
                stats["avg_over_parameters"][target_expectation]["grads_to_low_error"]
            )
            STEP_SIZE = STEP_SIZE.at[i].set(stats["step_size"])
        if opt == "max":
            iopt = np.argmin(Num_Grads)
        elif opt == "avg":
            iopt = np.argmin(Num_Grads_AVG)
        else:
            raise Exception("opt not recognized")
        edge = grid_iteration == 0 and (iopt == 0 or iopt == (len(Lgrid) - 1))

    return (
        Lgrid[iopt],
        STEP_SIZE[iopt],
        Num_Grads[iopt],
        Num_Grads_AVG[iopt],
        edge,
        inverse_mass_matrix,
        blackjax_state_after_tuning,
    )

def grid_search_adjusted_mclmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt=None,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    target_expectation=None,
    diagonal_preconditioning=True,
    acc_rate=0.99,
    grid_search_steps=None,
):
    
    def s(model, num_steps, initial_position, key):
        from sampler_comparison.experiments.utils import model_info
        
        # Get model-specific preferences if available
        model_name = model.name
        if model_name in model_info:
            model_prefs = model_info[model_name]
            # Use model-specific preferences if not explicitly provided
            if target_expectation is None:
                target_expectation = model_prefs.get('preferred_statistic', 'square')
            if opt is None:
                opt = "max" if model_prefs.get('max_over_parameters', True) else "avg"
            if grid_search_steps is None:
                grid_search_steps = model_prefs.get('grid_search_steps', num_steps // 10)
        else:
            # Fallback defaults if model not in model_info
            if target_expectation is None:
                target_expectation = 'square'
            if opt is None:
                opt = "max"
            if grid_search_steps is None:
                grid_search_steps = num_steps // 10

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=grid_search_steps,  # Use model-specific grid search steps
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='adjusted_mclmc',
            target_expectation=target_expectation,
            diagonal_preconditioning=diagonal_preconditioning,
            acc_rate=acc_rate,
        )

        sampler=adjusted_mclmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    L_proposal_factor=jnp.inf,
                    random_trajectory_length=True,
                    return_samples=return_samples,
                )
        
 

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


def grid_search_hmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt=None,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    L_proposal_factor=jnp.inf,
    diagonal_preconditioning=True,
    target_expectation=None,
    grid_search_steps=None,
):
    
    def s(model, num_steps, initial_position, key):
        from sampler_comparison.experiments.utils import model_info
        
        # Get model-specific preferences if available
        model_name = model.name
        if model_name in model_info:
            model_prefs = model_info[model_name]
            # Use model-specific preferences if not explicitly provided
            if target_expectation is None:
                target_expectation = model_prefs.get('preferred_statistic', 'square')
            if opt is None:
                opt = "max" if model_prefs.get('max_over_parameters', True) else "avg"
            if grid_search_steps is None:
                grid_search_steps = model_prefs.get('grid_search_steps', num_steps // 10)
        else:
            # Fallback defaults if model not in model_info
            if target_expectation is None:
                target_expectation = 'square'
            if opt is None:
                opt = "max"
            if grid_search_steps is None:
                grid_search_steps = num_steps // 10

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=grid_search_steps,  # Use model-specific grid search steps
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='adjusted_hmc',
            euclidean=True,
            L_proposal_factor=L_proposal_factor,
            diagonal_preconditioning=diagonal_preconditioning,
            target_expectation=target_expectation,
        )

        sampler=adjusted_hmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    random_trajectory_length=True,
                    return_samples=return_samples,
                    L_proposal_factor=L_proposal_factor,
                )

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s

def grid_search_unadjusted_lmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt=None,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=1e-4,
    diagonal_preconditioning=True,
    target_expectation=None,
    grid_search_steps=None,
):
    
    def s(model, num_steps, initial_position, key):
        from sampler_comparison.experiments.utils import model_info
        
        # Get model-specific preferences if available
        model_name = model.name
        if model_name in model_info:
            model_prefs = model_info[model_name]
            # Use model-specific preferences if not explicitly provided
            if target_expectation is None:
                target_expectation = model_prefs.get('preferred_statistic', 'square')
            if opt is None:
                opt = "max" if model_prefs.get('max_over_parameters', True) else "avg"
            if grid_search_steps is None:
                grid_search_steps = model_prefs.get('grid_search_steps', num_steps // 10)
        else:
            # Fallback defaults if model not in model_info
            if target_expectation is None:
                target_expectation = 'square'
            if opt is None:
                opt = "max"
            if grid_search_steps is None:
                grid_search_steps = num_steps // 10

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=grid_search_steps,  # Use model-specific grid search steps
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='unadjusted_lmc',
            euclidean=True,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
            target_expectation=target_expectation,
        )

        sampler=unadjusted_lmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    return_samples=return_samples,
                )
        
 
        

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


def grid_search_unadjusted_hmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt=None,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=1e-4,
    diagonal_preconditioning=True,
    target_expectation=None,
    grid_search_steps=None,
):
    
    def s(model, num_steps, initial_position, key):
        from sampler_comparison.experiments.utils import model_info
        
        # Get model-specific preferences if available
        model_name = model.name
        if model_name in model_info:
            model_prefs = model_info[model_name]
            # Use model-specific preferences if not explicitly provided
            if target_expectation is None:
                target_expectation = model_prefs.get('preferred_statistic', 'square')
            if opt is None:
                opt = "max" if model_prefs.get('max_over_parameters', True) else "avg"
            if grid_search_steps is None:
                grid_search_steps = model_prefs.get('grid_search_steps', num_steps // 10)
        else:
            # Fallback defaults if model not in model_info
            if target_expectation is None:
                target_expectation = 'square'
            if opt is None:
                opt = "max"
            if grid_search_steps is None:
                grid_search_steps = num_steps // 10

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=grid_search_steps,  # Use model-specific grid search steps
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='unadjusted_hmc',
            euclidean=True,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
            target_expectation=target_expectation,
        )

        sampler=unadjusted_hmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    return_samples=return_samples,
                )
        
 
        

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


