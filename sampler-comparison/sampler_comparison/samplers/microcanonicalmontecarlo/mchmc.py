import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.adaptation.unadjusted_alba import unadjusted_alba


def unadjusted_mchmc_no_tuning(
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

        alg = blackjax.mchmc(
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



def unadjusted_mchmc(
    diagonal_preconditioning=True,
    integrator_type="mclachlan",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=5e-4,
    return_only_final=False,
    incremental_value_transform=None,
    alba_factor=0.4,
):
    # jax.debug.print("unadjusted_mchmc {x}", x=alba_factor)
    def s(model, num_steps, initial_position, key):


        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        num_alba_steps = num_tuning_steps // 3
        warmup = unadjusted_alba(
            mcmc_kernel=blackjax.mchmc.build_kernel(map_integrator_type_to_integrator["mchmc"][integrator_type]),
            init=blackjax.mchmc.init,
            logdensity_fn=logdensity_fn, 
            target_eevpd=desired_energy_var, 
            v=1., 
            num_alba_steps=num_alba_steps,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            )
        
        (blackjax_state_after_tuning, blackjax_mchmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)

        num_tuning_integrator_steps = num_tuning_steps

        expectations, metadata = unadjusted_mchmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mchmc_sampler_params['step_size'],
            L=blackjax_mchmc_sampler_params['L'],
            inverse_mass_matrix=blackjax_mchmc_sampler_params['inverse_mass_matrix'],
            return_samples=return_samples,
            return_only_final=return_only_final,
            incremental_value_transform=incremental_value_transform,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s


def grid_search_unadjusted_mchmc(
    num_chains,
    integrator_type,
    grid_size=6,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=5e-4,
    diagonal_preconditioning=True,
    alba_factor=0.3,
    preferred_statistic=None,
    preferred_max_over_parameters=None,
    preferred_grid_search_steps=None,
):
    """
    Cleaner and more principled grid search for unadjusted MCHMC with ALBA warmup.
    
    Args:
        num_chains: Number of chains to run
        integrator_type: Type of integrator to use
        grid_size: Number of L values to try in each grid iteration
        grid_iterations: Number of grid search iterations
        num_tuning_steps: Number of tuning steps for ALBA warmup
        return_samples: Whether to return samples
        desired_energy_var: Desired energy variance for ALBA
        diagonal_preconditioning: Whether to use diagonal preconditioning
        alba_factor: Factor for ALBA adaptation
        preferred_statistic: Which statistic to optimize ("square", "abs", "entropy", etc.) - if None, will use model-specific preference
        preferred_max_over_parameters: Whether to use max_over_parameters (True) or avg_over_parameters (False) - if None, will use model-specific preference
        preferred_grid_search_steps: Number of steps for grid search evaluation - if None, will use model-specific preference
    
    Returns:
        A sampler function that can be used with the benchmark framework
    """
    
    def s(model, num_steps, initial_position, key):
        from sampler_comparison.samplers.grid_search.grid_search import grid_search_L
        from sampler_comparison.experiments.utils import get_model_specific_preferences
        
        # Get model-specific preferences
        statistic, max_over_parameters, grid_search_steps = get_model_specific_preferences(
            model, False, preferred_statistic, preferred_max_over_parameters, preferred_grid_search_steps, num_steps
        )
        
        tune_key, grid_key, run_key = jax.random.split(key[0], 3)
        
        print(f"\n=== ALBA Warmup (MCHMC) ===")
        print(f"Model: {model.name} (ndims={model.ndims})")
        print(f"Number of tuning steps: {num_tuning_steps}")
        print(f"Desired energy variance: {desired_energy_var}")
        print(f"Diagonal preconditioning: {diagonal_preconditioning}")
        print(f"ALBA factor: {alba_factor}")
        
        # ALBA warmup
        logdensity_fn = make_log_density_fn(model)
        num_alba_steps = num_tuning_steps // 3
        warmup = unadjusted_alba(
            mcmc_kernel=blackjax.mchmc.build_kernel(map_integrator_type_to_integrator["mchmc"][integrator_type]),
            init=blackjax.mchmc.init,
            logdensity_fn=logdensity_fn, 
            target_eevpd=desired_energy_var, 
            v=1., 
            num_alba_steps=num_alba_steps,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
        )
        
        (alba_state, alba_params), adaptation_info = warmup.run(tune_key, initial_position[0], num_tuning_steps)
        
        print(f"ALBA warmup complete!")
        print(f"  ALBA step size: {alba_params['step_size']:.6f}")
        print(f"  ALBA L: {alba_params['L']:.4f}")
        print(f"  ALBA inverse mass matrix shape: {alba_params['inverse_mass_matrix'].shape}")
        
        # Run the new grid search with ALBA state and parameters
        optimal_L, optimal_step_size, optimal_value, all_values, optimal_idx, tuning_outcome = grid_search_L(
            model=model,
            num_gradient_calls=grid_search_steps,  # Use model-specific grid search steps as gradient calls
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=grid_key,
            initial_L=alba_params['L'],
            initial_inverse_mass_matrix=alba_params['inverse_mass_matrix'],
            initial_step_size=alba_params['step_size'],
            initial_state=alba_state,
            sampler_fn=unadjusted_mchmc_no_tuning,
            algorithm=blackjax.mchmc,  # Pass algorithm directly
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],  # Pass integrator directly
            statistic=statistic,  # Use model-specific statistic
            max_over_parameters=max_over_parameters,  # Use model-specific parameter type
            grid_size=grid_size,
            grid_iterations=grid_iterations,
            is_adjusted_sampler=False,  # This is an unadjusted sampler
            desired_energy_var=desired_energy_var,  # For robnik_step_size_tuning
        )
        
        print(f"\n=== Final Sampling (MCHMC) ===")
        print(f"Using optimal L: {optimal_L:.4f}")
        print(f"Using optimal step_size: {optimal_step_size:.6f}")
        print(f"Using ALBA inverse mass matrix")
        print(f"Using statistic: {statistic}")
        print(f"Using {'max' if max_over_parameters else 'avg'} over parameters")
        print(f"Tuning outcome: {tuning_outcome}")
        
        # Create the final sampler with the optimal L and step_size
        sampler = unadjusted_mchmc_no_tuning(
            initial_state=alba_state,
            integrator_type=integrator_type,
            step_size=optimal_step_size,
            L=optimal_L,
            inverse_mass_matrix=alba_params['inverse_mass_matrix'],
            return_samples=return_samples,
        )
        
        # Create a wrapper that adds tuning outcome to metadata
        def sampler_with_tuning_outcome(model, num_steps, initial_position, key):
            expectations, metadata = sampler(model, num_steps, initial_position, key)
            # Add tuning outcome to metadata
            metadata = metadata | {"tuning_outcome": tuning_outcome}
            return expectations, metadata
        
        return jax.pmap(
            lambda key, pos: sampler_with_tuning_outcome(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(jax.random.split(run_key, num_chains), initial_position)
    
    return s
