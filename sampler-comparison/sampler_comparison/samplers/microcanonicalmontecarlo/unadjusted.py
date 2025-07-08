import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
    sampler_grads_to_low_error,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.unadjusted_alba import unadjusted_alba


def unadjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
):
    """
    Args:
        initial_state: Initial state of the chain
        integrator_type: Type of integrator to use (e.g. velocity verlet, mclachlan...)
        step_size: Step size to use
        L: Number of steps to run the chain for
        inverse_mass_matrix: Inverse mass matrix to use
        return_samples: Whether to return the samples or not
    Returns:
        A tuple of the form (expectations, stats) where expectations are the expectations of the chain and stats are the hyperparameters of the chain (L, stepsize and inverse mass matrix) and other metadata
    """


    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        alg = blackjax.mclmc(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )

        if return_samples:
            transform = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )

            get_final_sample = lambda state, info: (model.default_event_space_bijector(state.position), info)

            state = initial_state

        else:
            alg, init, transform = with_only_statistics(
                model=model,
                alg=alg,
                incremental_value_transform=incremental_value_transform,
            )

            state = init(initial_state)

            get_final_sample = lambda output: output[1][1]

        final_output, history = run_inference_algorithm(
            rng_key=key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=(lambda a, b: None) if return_only_final else transform,
            progress_bar=False,
        )


        if return_only_final:

            return get_final_sample(final_output, {})

        (expectations, info) = history

        return (
            expectations,
            {
                "L": L,
                "step_size": step_size,
                "acc_rate": jnp.nan,
                "num_tuning_grads": 0,
                "num_grads_per_proposal": calls_per_integrator_step(integrator_type),
            },
        )

    return s


def grid_search_2d(
    evaluate_fn,
    param1_name,
    param1_range,
    param2_name,
    param2_range,
    opt="min",
    **kwargs
):
    """
    Generic 2D grid search function.
    
    Args:
        evaluate_fn: Function that takes (param1, param2, **kwargs) and returns a score
        param1_name: Name of first parameter
        param1_range: Range of values for first parameter
        param2_name: Name of second parameter  
        param2_range: Range of values for second parameter
        opt: Optimization direction - "min" or "max"
        **kwargs: Additional arguments to pass to evaluate_fn
        
    Returns:
        Tuple of (optimal_param1, optimal_param2, optimal_score, all_results)
    """
    grid_size1 = len(param1_range)
    grid_size2 = len(param2_range)
    results = jnp.zeros((grid_size1, grid_size2))
    
    # Evaluate all combinations
    for i, param1 in enumerate(param1_range):
        for j, param2 in enumerate(param2_range):
            kwargs[param1_name] = param1
            kwargs[param2_name] = param2
            results = results.at[i, j].set(evaluate_fn(**kwargs))
    
    # Find optimal parameters
    if opt == "min":
        optimal_idx = jnp.unravel_index(jnp.argmin(results), results.shape)
    elif opt == "max":
        optimal_idx = jnp.unravel_index(jnp.argmax(results), results.shape)
    else:
        raise ValueError(f"Unknown optimization criterion: {opt}")
    
    optimal_param1 = param1_range[optimal_idx[0]]
    optimal_param2 = param2_range[optimal_idx[1]]
    optimal_score = results[optimal_idx[0], optimal_idx[1]]
    
    return optimal_param1, optimal_param2, optimal_score, results


def unadjusted_mclmc_grid_search(
    initial_state,
    integrator_type,
    inverse_mass_matrix,
    initial_L,
    initial_step_size,
    num_steps,
    num_chains,
    key,
    grid_size=10,
    opt="max",
    target_expectation='square',
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
):
    """
    Perform a 2D grid search over L and step_size parameters for unadjusted MCLMC.
    
    Args:
        initial_state: Initial state of the chain
        integrator_type: Type of integrator to use
        inverse_mass_matrix: Inverse mass matrix to use
        initial_L: Initial L value to center the grid around
        initial_step_size: Initial step size to center the grid around
        num_steps: Number of steps to run each sampler for
        num_chains: Number of chains to run in parallel
        key: Random key for reproducibility
        grid_size: Number of grid points for each parameter (default: 10)
        opt: Optimization criterion - "max" or "avg" (default: "max")
        target_expectation: Which expectation to optimize for (default: 'square')
        return_samples: Whether to return samples from the final sampler
        incremental_value_transform: Transform for incremental values
        return_only_final: Whether to return only final sample
        
    Returns:
        A sampler function configured with the optimal parameters
    """
    
    def grid_search_2d_sampler(model, num_steps, initial_position, key):
        """Main grid search function."""
        
        # Create grid ranges around initial values
        L_range = jnp.linspace(initial_L / 3, initial_L * 3, grid_size)
        step_size_range = jnp.linspace(initial_step_size / 3, initial_step_size * 3, grid_size)
        
        # Initialize arrays to store results
        results = jnp.zeros((grid_size, grid_size))
        
        # Split keys for each evaluation
        keys = jax.random.split(key, grid_size * grid_size)
        
        # Evaluate all combinations
        for i, L in enumerate(L_range):
            for j, step_size in enumerate(step_size_range):
                eval_key = keys[i * grid_size + j]
                
                # Create sampler for this parameter combination
                sampler = unadjusted_mclmc_no_tuning(
                    initial_state=initial_state,
                    integrator_type=integrator_type,
                    step_size=step_size,
                    L=L,
                    inverse_mass_matrix=inverse_mass_matrix,
                    return_samples=False,  # We don't need samples for evaluation
                    incremental_value_transform=incremental_value_transform,
                    return_only_final=return_only_final,
                )
                
                # Create a pmapped version of the sampler for parallel evaluation
                pmapped_sampler = jax.pmap(
                    lambda key, pos: sampler(
                        model=model, num_steps=num_steps, initial_position=pos, key=key
                    )
                )
                
                # Evaluate the sampler
                (stats, _) = sampler_grads_to_low_error(
                    model=model,
                    sampler=pmapped_sampler,
                    key=eval_key,
                    batch_size=num_chains,
                )
                
                results = results.at[i, j].set(
                    stats[f"{opt}_over_parameters"][target_expectation]["grads_to_low_error"]
                )
        
        # Find optimal parameters
        optimal_idx = jnp.unravel_index(jnp.argmin(results), results.shape)
        optimal_L = L_range[optimal_idx[0]]
        optimal_step_size = step_size_range[optimal_idx[1]]
        optimal_score = results[optimal_idx[0], optimal_idx[1]]
        
        # Create and return the optimal sampler
        optimal_sampler = unadjusted_mclmc_no_tuning(
            initial_state=initial_state,
            integrator_type=integrator_type,
            step_size=optimal_step_size,
            L=optimal_L,
            inverse_mass_matrix=inverse_mass_matrix,
            return_samples=return_samples,
            incremental_value_transform=incremental_value_transform,
            return_only_final=return_only_final,
        )
        
        return optimal_sampler(model, num_steps, initial_position, key)
    
    return grid_search_2d_sampler


# def unadjusted_mclmc_grid_search(integrator_type, num_steps, initial_position, key, return_samples, return_only_final, incremental_value_transform):

#     func = undefined

#     params = grid_search(func)

#     return unadjusted_mclmc_no_tuning(
#             initial_state=blackjax_state_after_tuning,
#             integrator_type=integrator_type,
#             step_size=params['step_size'],
#             L=params['L'],
#             inverse_mass_matrix=params['inverse_mass_matrix'],
#             return_samples=return_samples,
#             return_only_final=return_only_final,
#             incremental_value_transform=incremental_value_transform,
#         )(model, num_steps, initial_position, run_key)



def unadjusted_mclmc(
    diagonal_preconditioning=True,
    integrator_type="mclachlan",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=5e-4,
    return_only_final=False,
    incremental_value_transform=None,
    alba_factor=0.4,
    grid_search=False,
    grid_size=10,
    opt="max",
    target_expectation='square',
    num_chains=10,
):
    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        if grid_search:
            # Use grid search instead of ALBA adaptation
            # First, we need to get initial parameters for the grid search
            # We'll use some reasonable defaults based on the model
            initial_L = jnp.sqrt(model.ndims) if not diagonal_preconditioning else 1.0
            initial_step_size = 0.1
            inverse_mass_matrix = jnp.ones(model.ndims)
            
            # Create initial state for grid search
            initial_state = blackjax.mcmc.mclmc.init(
                position=initial_position,
                logdensity_fn=logdensity_fn,
                random_generator_arg=jax.random.key(0),
            )
            
            # Create grid search sampler
            grid_search_sampler = unadjusted_mclmc_grid_search(
                initial_state=initial_state,
                integrator_type=integrator_type,
                inverse_mass_matrix=inverse_mass_matrix,
                initial_L=initial_L,
                initial_step_size=initial_step_size,
                num_steps=num_steps,
                num_chains=num_chains,
                key=tune_key,
                grid_size=grid_size,
                opt=opt,
                target_expectation=target_expectation,
                return_samples=return_samples,
                incremental_value_transform=incremental_value_transform,
                return_only_final=return_only_final,
            )
            
            # Run the grid search and get results
            expectations, metadata = grid_search_sampler(
                model=model, 
                num_steps=num_steps, 
                initial_position=initial_position, 
                key=run_key
            )
            
            # Calculate tuning gradient calls (grid search evaluation)
            num_tuning_integrator_steps = grid_size * grid_size * num_steps * num_chains
            
        else:
            # Use ALBA adaptation as before
            num_alba_steps = num_tuning_steps // 3
            warmup = unadjusted_alba(
                algorithm=blackjax.mclmc, 
                logdensity_fn=logdensity_fn, integrator=map_integrator_type_to_integrator["mclmc"][integrator_type], 
                target_eevpd=desired_energy_var, 
                v=1., 
                num_alba_steps=num_alba_steps,
                preconditioning=diagonal_preconditioning,
                alba_factor=alba_factor,
                )
            
            (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)

            num_tuning_integrator_steps = num_tuning_steps

            expectations, metadata = unadjusted_mclmc_no_tuning(
                initial_state=blackjax_state_after_tuning,
                integrator_type=integrator_type,
                step_size=blackjax_mclmc_sampler_params['step_size'],
                L=blackjax_mclmc_sampler_params['L'],
                inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
                return_samples=return_samples,
                return_only_final=return_only_final,
                incremental_value_transform=incremental_value_transform,
            )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s
