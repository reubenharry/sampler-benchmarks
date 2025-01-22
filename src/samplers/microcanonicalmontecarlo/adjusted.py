from typing import Callable, Union
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size, store_only_expectation_values
from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_make_L_step_size_adaptation,
    adjusted_mclmc_make_adaptation_L,
)
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from jax.flatten_util import ravel_pytree

from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from src.samplers.general import with_only_statistics
from src.util import calls_per_integrator_step, map_integrator_type_to_integrator


def adjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    L_proposal_factor=jnp.inf,
    random_trajectory_length=True,
    return_samples=False,
):

    def s(model, num_steps, initial_position, key):

        num_steps_per_traj = L / step_size
        if random_trajectory_length:
            # Halton sequence
            integration_steps_fn = lambda k: jnp.ceil(
                jax.random.uniform(k) * rescale(num_steps_per_traj)
            )
        else:
            integration_steps_fn = lambda _: jnp.ceil(num_steps_per_traj)

        alg = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn=model.unnormalized_log_prob,
            step_size=step_size,
            integration_steps_fn=integration_steps_fn,
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
            inverse_mass_matrix=inverse_mass_matrix,
            L_proposal_factor=L_proposal_factor,
        )

        fast_key, slow_key = jax.random.split(key, 2)

        if return_samples:
            (expectations, info) = run_inference_algorithm(
                rng_key=slow_key,
                initial_state=initial_state,
                inference_algorithm=alg,
                num_steps=num_steps,
                transform=lambda state, info: (
                    (model.default_event_space_bijector(state.position), info)
                ),
                progress_bar=False,
            )[1]

        else:
            results = with_only_statistics(
                model=model,
                alg=alg,
                initial_state=initial_state,
                rng_key=fast_key,
                num_steps=num_steps,
            )
            expectations, info = results[0], results[1]

        results = with_only_statistics(model, alg, initial_state, fast_key, num_steps)
        expectations, info = results[0], results[1]

        return (
            expectations,
            MCLMCAdaptationState(
                L=L, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
            ),
            info,
        )

    return s


def adjusted_mclmc_tuning(
    initial_position,
    num_steps,
    rng_key,
    logdensity_fn,
    diagonal_preconditioning,
    target_acc_rate,
    random_trajectory_length,
    integrator,
    L_proposal_factor,
    params=None,
    max="avg",
    num_windows=1,
    tuning_factor=1.0,
    num_tuning_steps=500,
):

    init_key, tune_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    frac_tune1 = num_tuning_steps / (3 * num_steps)
    frac_tune2 = num_tuning_steps / (3 * num_steps)
    frac_tune3 = num_tuning_steps / (3 * num_steps)

    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        )
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
            avg_num_integration_steps
        )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        inverse_mass_matrix=inverse_mass_matrix,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        L_proposal_factor=L_proposal_factor,
    )

    return blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        frac_tune3=frac_tune3,
        diagonal_preconditioning=diagonal_preconditioning,
        params=params,
        max=max,
        num_windows=num_windows,
        tuning_factor=tuning_factor,
    )


def adjusted_mclmc(
    integrator_type="velocity_verlet",
    diagonal_preconditioning=True,
    L_proposal_factor=jnp.inf,
    target_acc_rate=0.9,
    initial_params=None,
    max="avg",
    num_windows=2,
    random_trajectory_length=True,
    tuning_factor=1.0,
    num_tuning_steps=5000,
    return_samples=False,
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
    Returns:
    A function that runs the adjusted MCLMC sampler
    """

    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_integrator_steps,
        ) = adjusted_mclmc_tuning(
            initial_position=initial_position,
            num_steps=num_steps,
            rng_key=tune_key,
            logdensity_fn=model.unnormalized_log_prob,
            diagonal_preconditioning=diagonal_preconditioning,
            target_acc_rate=target_acc_rate,
            random_trajectory_length=random_trajectory_length,
            integrator=integrator,
            L_proposal_factor=L_proposal_factor,
            params=initial_params,
            max=max,
            num_windows=num_windows,
            tuning_factor=tuning_factor,
            num_tuning_steps=num_tuning_steps,
        )

        expectations, params, info = adjusted_mclmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mclmc_sampler_params.step_size,
            L=blackjax_mclmc_sampler_params.L,
            inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
            L_proposal_factor=L_proposal_factor,
            random_trajectory_length=random_trajectory_length,
            return_samples=return_samples,
        )(model, num_steps, initial_position, run_key)

        num_steps_per_traj = params.L / params.step_size

        return expectations, {
            "L": params.L,
            "step_size": params.step_size,
            "acc_rate": info.acceptance_rate.mean(),
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type),
            "num_grads_per_proposal": num_steps_per_traj
            * calls_per_integrator_step(integrator_type),
        }

    return s
