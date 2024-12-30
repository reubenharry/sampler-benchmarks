from typing import Callable
from chex import PRNGKey
import jax
import jax.numpy as jnp
import blackjax

# from blackjax.adaptation.window_adaptation import da_adaptation

from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.util import pytree_size, store_only_expectation_values
from blackjax.adaptation.step_size import (
    dual_averaging_adaptation,
)
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from src.util import *


def da_adaptation(
    rng_key: PRNGKey,
    initial_position,
    algorithm,
    logdensity_fn: Callable,
    num_steps: int = 1000,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    integrator=blackjax.mcmc.integrators.velocity_verlet,
):

    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    kernel = algorithm.build_kernel(integrator=integrator)
    init_kernel_state = algorithm.init(initial_position, logdensity_fn)
    inverse_mass_matrix = jnp.ones(pytree_size(initial_position))

    def step(state, key):

        adaptation_state, kernel_state = state

        new_kernel_state, info = kernel(
            key,
            kernel_state,
            logdensity_fn,
            jnp.exp(adaptation_state.log_step_size),
            inverse_mass_matrix,
        )

        new_adaptation_state = da_update(
            adaptation_state,
            info.acceptance_rate,
        )

        return (
            (new_adaptation_state, new_kernel_state),
            (True),
        )

    keys = jax.random.split(rng_key, num_steps)
    init_state = da_init(initial_step_size), init_kernel_state
    (adaptation_state, kernel_state), _ = jax.lax.scan(
        step,
        init_state,
        keys,
    )
    return kernel_state, {
        "step_size": da_final(adaptation_state),
        "inverse_mass_matrix": inverse_mass_matrix,
    }


# produce a kernel that only stores the average values of the bias for E[x_2] and Var[x_2]
def with_only_statistics(
    model, alg, initial_state, key, num_steps, incremental_value_transform=None
):

    # TODO: this is x not x**2!!!
    if incremental_value_transform is None:
        incremental_value_transform = lambda x: jnp.array(
            [
                jnp.average(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "identity"
                    ].ground_truth_standard_deviation
                ),
                # jnp.sqrt(jnp.average(jnp.square(x - model.sample_transformations['identity'].ground_truth_mean) / model.sample_transformations['identity'].ground_truth_standard_deviation)),
                # jnp.sqrt(jnp.average(jnp.square(x - model.sample_transformations['identity'].ground_truth_mean) / (model.sample_transformations['identity'].ground_truth_standard_deviation))),
                jnp.max(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "identity"
                    ].ground_truth_standard_deviation
                ),
            ]
        )

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: jnp.array(
            [
                model.sample_transformations["identity"](state.position) ** 2,
                model.sample_transformations["identity"](state.position),
            ]
        ),
        incremental_value_transform=incremental_value_transform,
    )

    return run_inference_algorithm(
        rng_key=key,
        initial_state=memory_efficient_sampling_alg.init(initial_state),
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )[1]


def nuts(
    integrator_type="velocity_verlet",
    preconditioning=True,
    return_ess_corr=False,
    return_samples=False,
    incremental_value_transform=None,
):
    def s(model, num_steps, initial_position, key):
        # num_tuning_steps = num_steps // 5
        num_tuning_steps = 2000

        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        rng_key, warmup_key = jax.random.split(key, 2)

        if not preconditioning:
            state, params = da_adaptation(
                rng_key=warmup_key,
                initial_position=initial_position,
                algorithm=blackjax.nuts,
                integrator=integrator,
                logdensity_fn=model.unnormalized_log_prob,
                num_steps=num_tuning_steps,
                target_acceptance_rate=0.8,
            )

        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts, model.unnormalized_log_prob, integrator=integrator
            )
            (state, params), _ = warmup.run(
                warmup_key, initial_position, num_tuning_steps
            )

        alg = blackjax.nuts(
            logdensity_fn=model.unnormalized_log_prob,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
        )

        fast_key, slow_key = jax.random.split(rng_key, 2)

        results = with_only_statistics(
            model,
            alg,
            state,
            fast_key,
            num_steps,
            incremental_value_transform=incremental_value_transform,
        )
        expectations, info = results[0], results[1]

        ess_corr = jax.lax.cond(
            not return_ess_corr,
            lambda: jnp.inf,
            lambda: jnp.mean(
                effective_sample_size(
                    jax.vmap(lambda x: ravel_pytree(x)[0])(
                        run_inference_algorithm(
                            rng_key=slow_key,
                            initial_state=state,
                            inference_algorithm=alg,
                            num_steps=num_steps,
                            transform=lambda state, _: (
                                model.sample_transformations["identity"](state.position)
                            ),  # TODO: transform?
                            progress_bar=False,
                        )[1]
                    )[None, ...]
                )
            )
            / num_steps,
        )

        if return_samples:
            expectations = run_inference_algorithm(
                rng_key=slow_key,
                initial_state=state,
                inference_algorithm=alg,
                num_steps=num_steps,
                transform=lambda state, _: (
                    model.sample_transformations["identity"](state.position)
                ),
                progress_bar=False,
            )[1]

        return (
            expectations,
            {
                "params": params,
                "num_grads_per_proposal": info.num_integration_steps.mean()
                * calls_per_integrator_step(integrator_type),
                "acc_rate": info.acceptance_rate.mean(),
                "ess_corr": ess_corr,
                "num_tuning_steps": num_tuning_steps,
            },
        )

    return s


samplers = {
    "nuts": nuts,
}
