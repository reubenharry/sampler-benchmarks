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
from blackjax.adaptation.unadjusted_alba import unadjusted_alba


def unadjusted_hmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
    desired_energy_var=3e-4,
    desired_energy_var_max_ratio=1e3,
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

        alg = blackjax.uhmc(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
            desired_energy_var_max_ratio=desired_energy_var_max_ratio,
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


def unadjusted_hmc(
    diagonal_preconditioning=True,
    integrator_type="velocity_verlet",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=3e-4,
    desired_energy_var_max_ratio=1e3,
    return_only_final=False,
    incremental_value_transform=None,
    alba_factor=0.4,
):
    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)
        tune_key, run_key = jax.random.split(key, 2)
        num_dimensions = initial_position.shape[0]

        num_alba_steps = num_tuning_steps // 3
        warmup = unadjusted_alba(
            algorithm=blackjax.uhmc, 
            logdensity_fn=logdensity_fn, integrator=map_integrator_type_to_integrator["hmc"][integrator_type], 
            target_eevpd=desired_energy_var, 
            v=jnp.sqrt(num_dimensions), num_alba_steps=num_alba_steps,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            )
        num_tuning_integrator_steps = num_tuning_steps
        
        (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)

        expectations, metadata = unadjusted_hmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mclmc_sampler_params['step_size'],
            L=blackjax_mclmc_sampler_params['L'],
            inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
            return_samples=return_samples,
            return_only_final=return_only_final,
            desired_energy_var=desired_energy_var,
            desired_energy_var_max_ratio=desired_energy_var_max_ratio,
            incremental_value_transform=incremental_value_transform,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s
