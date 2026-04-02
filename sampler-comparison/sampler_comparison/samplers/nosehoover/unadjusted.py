import sys
# import os
# print(os.listdir("../../blackjax"))
# raise Exception("stop")
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
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
    Q=None,
):
    """
    Unadjusted Nose-Hoover sampler (no tuning). Uses blackjax.nosehoover.

    Args:
        initial_state: Initial state of the chain (from blackjax.nosehoover.init).
        integrator_type: Kept for API consistency with other samplers; not used by NH.
        step_size: Step size to use.
        L: Kept for API consistency; each chain step is one NH integration step.
        inverse_mass_matrix: Inverse mass matrix to use.
        return_samples: Whether to return the samples or not.
        incremental_value_transform: Optional transform for expectation accumulation.
        return_only_final: If True, return only the final sample (and info).
        Q: Thermostat mass for Nose-Hoover; if None, blackjax uses g (ndim).
    Returns:
        A tuple of the form (expectations, stats) where expectations are the
        expectations of the chain and stats are hyperparameters and metadata.
    """

    def s(model, num_steps, initial_position, key):
        logdensity_fn = make_log_density_fn(model)

        alg = blackjax.nosehoover(
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            Q=Q,
        )

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
        # Each kernel step is one Nose-Hoover integration step = one gradient.
        num_grads_per_proposal = 1

        return (
            expectations,
            {
                "L": L,
                "step_size": step_size,
                "acc_rate": jnp.nan,
                "num_tuning_grads": 0,
                "num_grads_per_proposal": num_grads_per_proposal,
                "inverse_mass_matrix": inverse_mass_matrix,
                "info": info,
            },
        )

    return s


if __name__ == "__main__":
    """Run a short Nose-Hoover chain to check the sampler runs."""
    from collections import namedtuple

    from sampler_comparison.samplers.general import initialize_model

    # Minimal model for testing: standard normal in 2d (avoids sampler_evaluation deps)
    Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations"])
    log_density_fn = lambda z: -0.5 * jnp.sum(z**2)
    model = Model(
        ndims=2,
        log_density_fn=log_density_fn,
        default_event_space_bijector=lambda x: x,
        sample_transformations={},  # with_only_statistics may expect this
    )

    key = jax.random.PRNGKey(0)
    initial_position = initialize_model(model, key)
    key, init_key = jax.random.split(key)

    alg = blackjax.nosehoover(
        logdensity_fn=make_log_density_fn(model),
        step_size=0.1,
        inverse_mass_matrix=1.0,
    )
    initial_state = alg.init(initial_position, init_key)

    sampler = unadjusted_nosehoover_no_tuning(
        initial_state=initial_state,
        integrator_type="velocity_verlet",  # unused, for API consistency
        step_size=0.1,
        L=4,
        inverse_mass_matrix=1.0,
    )

    key, run_key = jax.random.split(key)
    num_steps = 200
    expectations, meta = sampler(model, num_steps, initial_position, run_key)

    print("Nose-Hoover sampler run OK")
    print("  num_steps:", num_steps)
    print("  L:", meta["L"], "step_size:", meta["step_size"])
    print("  num_grads_per_proposal:", meta["num_grads_per_proposal"])
    print("  expectations keys:", list(expectations.keys()) if hasattr(expectations, "keys") else "N/A")
