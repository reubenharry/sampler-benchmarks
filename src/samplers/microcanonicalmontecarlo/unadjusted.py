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

from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from src.samplers.general import with_only_statistics
from src.util import calls_per_integrator_step, map_integrator_type_to_integrator


def unadjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    sqrt_diag_cov,
    num_tuning_steps,
    return_ess_corr=False,
    return_samples=False,
):
    def s(model, num_steps, initial_position, key):

        fast_key, slow_key = jax.random.split(key, 2)

        alg = blackjax.mclmc(
            model.unnormalized_log_prob,
            L=L,
            step_size=step_size,
            sqrt_diag_cov=sqrt_diag_cov,
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )

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
                model, alg, initial_state, fast_key, num_steps
            )
            expectations, info = results[0], results[1]


        return (
            expectations,
            {
                "params": MCLMCAdaptationState(
                    L=L, step_size=step_size, sqrt_diag_cov=sqrt_diag_cov
                ),
                "num_grads_per_proposal": calls_per_integrator_step(integrator_type),
                "acc_rate": 1.0,
                "ess_corr": 0.0,
                "num_tuning_steps": num_tuning_steps,
            },
        )


    return s


def unadjusted_mclmc_tuning(
    initial_position,
    num_steps,
    rng_key,
    logdensity_fn,
    integrator_type,
    diagonal_preconditioning,
    frac_tune3=0.1,
    num_tuning_steps=500,
):

    tune_key, init_key = jax.random.split(rng_key, 2)

    frac_tune1 = num_tuning_steps / (2 * num_steps)
    frac_tune2 = num_tuning_steps / (2 * num_steps)

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
    )

    kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        sqrt_diag_cov=sqrt_diag_cov,
    )

    return blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=diagonal_preconditioning,
        frac_tune3=frac_tune3,
        frac_tune2=frac_tune2,
        frac_tune1=frac_tune1,
    )


def unadjusted_mclmc(
    preconditioning=True,
    integrator_type="mclachlan",
    frac_tune3=0.1,
    return_ess_corr=False,
    num_tuning_steps=500,
    return_samples=False,
):
    def s(model, num_steps, initial_position, key):

        tune_key, run_key = jax.random.split(key, 2)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = unadjusted_mclmc_tuning(
            initial_position,
            num_steps,
            tune_key,
            model.unnormalized_log_prob,
            integrator_type,
            preconditioning,
            frac_tune3,
            num_tuning_steps=num_tuning_steps,
        )

        # num_tuning_steps = (0.1 + 0.1) * num_windows * num_steps + frac_tune3 * num_steps

        return unadjusted_mclmc_no_tuning(
            blackjax_state_after_tuning,
            integrator_type,
            blackjax_mclmc_sampler_params.step_size,
            blackjax_mclmc_sampler_params.L,
            blackjax_mclmc_sampler_params.sqrt_diag_cov,
            num_tuning_steps,
            return_ess_corr=return_ess_corr,
            return_samples=return_samples,
        )(model, num_steps, initial_position, run_key)

    return s
