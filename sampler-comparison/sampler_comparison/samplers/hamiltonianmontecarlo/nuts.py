import jax
from blackjax.util import run_inference_algorithm
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
from sampler_comparison.util import *
import time

def nuts(
    integrator_type="velocity_verlet",
    diagonal_preconditioning=True,
    return_samples=False,
    incremental_value_transform=None,
    num_tuning_steps=5000,
    return_only_final=False,
    target_acc_rate=0.8,
    progress_bar=False,
    initial_state=None,
    initial_step_size=None,
    initial_inverse_mass_matrix=None,   
):

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)
        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        rng_key, warmup_key = jax.random.split(key, 2)

        if initial_state is None:

            warmup = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, integrator=integrator, 
                    preconditioning=diagonal_preconditioning,
                    target_acceptance_rate=target_acc_rate,
                    # adaptation_info_fn=lambda a , info, b : info,
                    adaptation_info_fn=lambda a , b, c : None,
                )
            (state, params), adaptation_info = warmup.run(
                    warmup_key, initial_position, num_tuning_steps
                )

        else:
            state = initial_state
            params = {'step_size': initial_step_size, 'inverse_mass_matrix': initial_inverse_mass_matrix}

        # save params and state
        


        # jax.debug.print("warmup finished")

        # adaptation_info = adaptation_info.info

        alg = blackjax.nuts(
            logdensity_fn=logdensity_fn,
            step_size=params["step_size"],
            inverse_mass_matrix=params["inverse_mass_matrix"],
            integrator=integrator,
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

        # JAX-compatible timing using debug callbacks
        # start_time = time.perf_counter()
        # jax.debug.print("Starting run_inference_algorithm at: {x}", x=start_time)
        
        final_output, history = run_inference_algorithm(
            rng_key=rng_key,
            initial_state=state,
            inference_algorithm=alg,
            num_steps=num_steps,
            transform=(lambda a, b: None) if return_only_final else transform,
            #progress_bar=progress_bar,
        )
        
        # Block until ready to ensure computation is complete
        # final_output[0].block_until_ready()
        # final_output[1].block_until_ready()
        # if not return_only_final:
        #     history.block_until_ready()
        
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # jax.debug.print("run_inference_algorithm completed in {x} seconds", x=elapsed_time)

        if return_only_final:

            return get_final_sample(final_output, {})

        (expectations, info) = history


        return (
            expectations,
            {
                "L": params["step_size"] * info.num_integration_steps.mean(),
                "step_size": params["step_size"],
                "num_grads_per_proposal": info.num_integration_steps.mean() * calls_per_integrator_step(integrator_type),
                # "acc_rate": info.acceptance_rate.mean(),
                # "num_tuning_grads": adaptation_info.num_integration_steps.sum()
                # * calls_per_integrator_step(integrator_type),
                'num_tuning_grads': 0,
                # 'num_grads_per_proposal': params['L'] / params['step_size'],
            },
        )

    return s
