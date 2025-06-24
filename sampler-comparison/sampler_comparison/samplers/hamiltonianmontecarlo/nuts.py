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
    return_only_final=False,
    target_acc_rate=0.8,
    # cos_angle_termination=0.,
    progress_bar=False,
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
                target_acceptance_rate=target_acc_rate,
                # cos_angle_termination=cos_angle_termination,
            )

        else:
            warmup = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn, integrator=integrator,
                #  cos_angle_termination=cos_angle_termination
            )
            (state, params), adaptation_info = warmup.run(
                warmup_key, initial_position, 20000
            )

            adaptation_info = adaptation_info.info

        jax.debug.print("params: {params}", params=params)

        alg = blackjax.nuts(
            logdensity_fn=logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
            # cos_angle_termination=cos_angle_termination,
        )

        if return_samples:
            transform = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )

            get_final_sample = lambda state, info: (model.default_event_space_bijector(state.position), info)

        else:
            alg, init, transform = with_only_statistics(
                model=model,
                alg=alg,
                incremental_value_transform=incremental_value_transform,
            )

            state = init(state)

            get_final_sample = lambda output, info: (output[1][1], info)

        final_output, history = run_inference_algorithm(
            rng_key=rng_key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=(lambda a, b: None) if return_only_final else transform,
            progress_bar=progress_bar,
        )

        if return_only_final:

            return get_final_sample(final_output, {})

        (expectations, info) = history

        return (
            expectations,
            {
                "L": params["step_size"] * info.num_integration_steps.mean(),
                "step_size": params["step_size"],
                "num_grads_per_proposal": info.num_integration_steps.mean()
                * calls_per_integrator_step(integrator_type),
                "acc_rate": info.acceptance_rate.mean(),
                "num_tuning_grads": adaptation_info.num_integration_steps.sum()
                * calls_per_integrator_step(integrator_type),
            },
        )

    return s
