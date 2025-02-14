import jax
from blackjax.util import run_inference_algorithm
import blackjax
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts_tuning import da_adaptation
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
from sampler_comparison.util import *


def nuts(
    integrator_type="velocity_verlet",
    diagonal_preconditioning=True,
    return_samples=False,
    incremental_value_transform=None,
    num_tuning_steps=5000,
):

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)
        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        rng_key, warmup_key = jax.random.split(key, 2)

        if not diagonal_preconditioning:
            state, params, adaptation_info = da_adaptation(
                rng_key=warmup_key,
                initial_position=initial_position,
                algorithm=blackjax.nuts,
                integrator=integrator,
                logdensity_fn=logdensity_fn,
                num_steps=num_tuning_steps,
                target_acceptance_rate=0.8,
            )

        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn, integrator=integrator
            )
            (state, params), adaptation_info = warmup.run(
                warmup_key, initial_position, num_tuning_steps
            )
            info = adaptation_info

        alg = blackjax.nuts(
            logdensity_fn=logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
        )

        fast_key, slow_key = jax.random.split(rng_key, 2)

        if return_samples:
            (expectations, info) = run_inference_algorithm(
                rng_key=slow_key,
                initial_state=state,
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
                initial_state=state,
                rng_key=fast_key,
                num_steps=num_steps,
                incremental_value_transform=incremental_value_transform,
            )
            expectations, info = results[0], results[1]

        return (
            expectations,
            {
                "L": params["step_size"] * info.num_integration_steps.mean(),
                "step_size": params["step_size"],
                "num_grads_per_proposal": info.num_integration_steps.mean()
                * calls_per_integrator_step(integrator_type),
                "acc_rate": info.acceptance_rate.mean(),
                "num_tuning_grads": info.num_integration_steps.sum()
                * calls_per_integrator_step(integrator_type),
                "num_grads_per_proposal": info.num_integration_steps.mean(),
            },
        )

    return s
