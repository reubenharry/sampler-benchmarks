import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
import blackjax

from blackjax.mcmc.adjusted_mclmc_dynamic import rescale, make_random_trajectory_length_fn
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.adjusted_alba import adjusted_alba


def adjusted_hmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    random_trajectory_length=True,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
    L_proposal_factor=jnp.inf,
):
    
    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        num_steps_per_traj = L / step_size
        

        integration_steps_fn = make_random_trajectory_length_fn(random_trajectory_length)

        alg = blackjax.dynamic_malt(
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            integration_steps_fn=integration_steps_fn(num_steps_per_traj),
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            inverse_mass_matrix=inverse_mass_matrix,
            L_proposal_factor=L_proposal_factor,
        )

        if return_samples:
            transform = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )

            get_final_sample = lambda _: None

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

            return get_final_sample(final_output)

        (expectations, info) = history

        return (
            expectations,
            {
                "L": L,
                "step_size": step_size,
                "acc_rate": info.acceptance_rate.mean(),
                "num_grads_per_proposal": jnp.clip(L
                / step_size, min=1)
                * calls_per_integrator_step(integrator_type),
                "num_tuning_grads": 0,
            },
        )

    return s




def adjusted_hmc(
    integrator_type="velocity_verlet",
    diagonal_preconditioning=True,
    target_acc_rate=0.9,
    random_trajectory_length=True,
    num_tuning_steps=20000,
    return_samples=False,
    return_only_final=False,
    L_proposal_factor=jnp.inf,
    incremental_value_transform=None,
    alba_factor=0.23,
):
    """
    Args:
    integrator_type: the integrator to use (e.g. 'mclachlan')
    diagonal_preconditioning: whether to use diagonal preconditioning
    L_proposal_factor: the factor to multiply L by to get the L value used in the proposal (infinite L means no Langevin noise in proposal)
    target_acc_rate: the target acceptance rate
    params: the initial parameters to use in the adaptation. If None, default parameters are used
    max: the method to use to calculate L in the first stage of adaptation for L
    num_windows: the number of windows to use in the adaptation
    random_trajectory_length: whether to use random trajectory length
    tuning_factor: the factor to multiply L by in the first stage of adaptation for L
    num_tuning_steps: the number of tuning steps to use
    L_factor_stage_3: the factor to multiply L by in the third stage of adaptation for L
    return_samples: whether to return samples
    warmup: whether to do a NUTS warmup or a warmup with MCLMC
    Returns:
    A function that runs the sampler
    """

    def s(model, num_steps, initial_position, key):


        logdensity_fn = make_log_density_fn(model)
        tune_key, run_key = jax.random.split(key, 2)
        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        num_dimensions = initial_position.shape[0]


        num_alba_steps = num_tuning_steps // 3

        warmup = adjusted_alba(
            unadjusted_algorithm=blackjax.langevin,
            logdensity_fn=logdensity_fn,
            target_eevpd=3e-4,
            num_alba_steps=num_alba_steps,
            v=jnp.sqrt(num_dimensions),
            adjusted_algorithm=blackjax.dynamic_malt,
            target_acceptance_rate=target_acc_rate,
            integrator=integrator,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            L_proposal_factor=L_proposal_factor,
        )

        (alba_state, alba_params, adaptation_info) = warmup.run(tune_key, initial_position, num_tuning_steps)
        
        num_tuning_integrator_steps = jnp.nan # adaptation_info.num_integration_steps.sum()

        expectations, metadata = adjusted_hmc_no_tuning(
            initial_state=alba_state,
            integrator_type=integrator_type,
            step_size=alba_params['step_size'],
            L=alba_params['L'],
            inverse_mass_matrix=alba_params['inverse_mass_matrix'],
            random_trajectory_length=random_trajectory_length,
            return_samples=return_samples,
            return_only_final=return_only_final,
            L_proposal_factor=L_proposal_factor,
            incremental_value_transform=incremental_value_transform,
        )(model, num_steps, initial_position, run_key)


        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s


def grid_search_adjusted_hmc(
    num_chains,
    integrator_type,
    grid_size=6,
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=3e-4,
    diagonal_preconditioning=True,
    alba_factor=0.23,
    preferred_statistic=None,
    preferred_max_over_parameters=None,
    preferred_grid_search_steps=None,
    L_proposal_factor=jnp.inf,
    target_acc_rate=0.9,
    random_trajectory_length=True,
):
    # raise Exception("stop")
    """
    Cleaner and more principled grid search for adjusted HMC with ALBA warmup.
    
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
        L_proposal_factor: Factor for L proposal in adjusted HMC
        target_acc_rate: Target acceptance rate for adjusted HMC
        random_trajectory_length: Whether to use random trajectory length
    
    Returns:
        A sampler function that can be used with the benchmark framework
    """
    
    def s(model, num_steps, initial_position, key):
        # jax.debug.print("foo bar {x}", x=key)
        # raise Exception("stop")
        from sampler_comparison.samplers.grid_search.grid_search import grid_search_L
        from sampler_comparison.experiments.utils import get_model_specific_preferences
        
        # Get model-specific preferences
        statistic, max_over_parameters, grid_search_steps = get_model_specific_preferences(
            model, True, preferred_statistic, preferred_max_over_parameters, preferred_grid_search_steps, num_steps
        )
        
        tune_key, grid_key, run_key = jax.random.split(key[0], 3)
        
        print(f"\n=== ALBA Warmup (Adjusted HMC) ===")
        print(f"Model: {model.name} (ndims={model.ndims})")
        print(f"Number of tuning steps: {num_tuning_steps}")
        print(f"Desired energy variance: {desired_energy_var}")
        print(f"Diagonal preconditioning: {diagonal_preconditioning}")
        print(f"ALBA factor: {alba_factor}")
        print(f"Target acceptance rate: {target_acc_rate}")
        print(f"L proposal factor: {L_proposal_factor}")
        
        # ALBA warmup
        logdensity_fn = make_log_density_fn(model)
        num_dimensions = initial_position[0].shape[0]
        num_alba_steps = num_tuning_steps // 3
        warmup = adjusted_alba(
            unadjusted_algorithm=blackjax.langevin,
            logdensity_fn=logdensity_fn,
            target_eevpd=desired_energy_var,
            num_alba_steps=num_alba_steps,
            v=jnp.sqrt(num_dimensions),
            adjusted_algorithm=blackjax.dynamic_malt,
            target_acceptance_rate=target_acc_rate,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            L_proposal_factor=L_proposal_factor,
        )
        
        (alba_state, alba_params, adaptation_info) = warmup.run(tune_key, initial_position[0], num_tuning_steps)
        
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
            sampler_fn=adjusted_hmc_no_tuning,
            algorithm=blackjax.dynamic_malt,  # Pass algorithm directly
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],  # Pass integrator directly
            statistic=statistic,  # Use model-specific statistic
            max_over_parameters=max_over_parameters,  # Use model-specific parameter type
            grid_size=5,
            grid_iterations=grid_iterations,
            is_adjusted_sampler=True,  # This is an adjusted sampler
            target_acc_rate=target_acc_rate,  # For da_adaptation
            L_proposal_factor=L_proposal_factor,  # For da_adaptation
            random_trajectory_length=random_trajectory_length,  # For da_adaptation
        )
        
        print(f"\n=== Final Sampling (Adjusted HMC) ===")
        print(f"Using optimal L: {optimal_L:.4f}")
        print(f"Using optimal step_size: {optimal_step_size:.6f}")
        print(f"Using ALBA inverse mass matrix")
        print(f"Using statistic: {statistic}")
        print(f"Using {'max' if max_over_parameters else 'avg'} over parameters")
        print(f"Tuning outcome: {tuning_outcome}")
        
        # Create the final sampler with the optimal L and step_size
        sampler = adjusted_hmc_no_tuning(
            initial_state=alba_state,
            integrator_type=integrator_type,
            step_size=optimal_step_size,
            L=optimal_L,
            inverse_mass_matrix=alba_params['inverse_mass_matrix'],
            random_trajectory_length=random_trajectory_length,
            return_samples=return_samples,
            L_proposal_factor=L_proposal_factor,
        )

        jax.debug.print("tuning_outcome {x}", x=tuning_outcome)
        
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


