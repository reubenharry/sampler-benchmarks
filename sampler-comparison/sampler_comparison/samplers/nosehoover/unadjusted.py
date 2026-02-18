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


def unadjusted_nosehoover_no_tuning(
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
    Unadjusted Nose-Hoover sampler (no tuning). The blackjax.nosehoover algorithm
    is not yet implemented; this is a stub that leaves that part incomplete.

    Args:
        initial_state: Initial state of the chain
        integrator_type: Type of integrator to use for the underlying dynamics
        step_size: Step size to use
        L: Number of steps per trajectory (or chain length parameter)
        inverse_mass_matrix: Inverse mass matrix to use
        return_samples: Whether to return the samples or not
        incremental_value_transform: Optional transform for expectation accumulation
        return_only_final: If True, return only the final sample (and info)
    Returns:
        A tuple of the form (expectations, stats) where expectations are the
        expectations of the chain and stats are hyperparameters and metadata.
    """

    def s(model, num_steps, initial_position, key):
        logdensity_fn = make_log_density_fn(model)

        # TODO: implement blackjax.nosehoover and use it here. For example:
        # alg = blackjax.nosehoover(
        #     logdensity_fn=logdensity_fn,
        #     L=L,
        #     step_size=step_size,
        #     inverse_mass_matrix=inverse_mass_matrix,
        #     integrator=map_integrator_type_to_integrator["nosehoover"][integrator_type],
        # )
        alg = None  # placeholder until blackjax.nosehoover is implemented

        if return_samples:
            transform = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )
            get_final_sample = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )
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
                "inverse_mass_matrix": inverse_mass_matrix,
                "info": info,
            },
        )

    return s
