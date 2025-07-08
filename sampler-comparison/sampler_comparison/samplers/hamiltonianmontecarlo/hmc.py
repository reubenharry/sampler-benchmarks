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
from blackjax.adaptation.adjusted_abla import alba_adjusted


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
                "num_grads_per_proposal": L
                / step_size
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

        warmup = alba_adjusted(
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

        blackjax_state_after_tuning, sampler_params, adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)
        
        num_tuning_integrator_steps = adaptation_info.num_integration_steps.sum()

        expectations, metadata = adjusted_hmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=sampler_params['step_size'],
            L=sampler_params['L'],
            inverse_mass_matrix=sampler_params['inverse_mass_matrix'],
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


