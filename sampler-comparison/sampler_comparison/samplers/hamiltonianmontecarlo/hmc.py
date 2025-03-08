import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
import blackjax
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_find_L_and_step_size,
    adjusted_mclmc_make_L_step_size_adaptation,
    adjusted_mclmc_make_adaptation_L,
)
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc_tuning,
)
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState



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
):
    
    print("Begin")

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        num_steps_per_traj = L / step_size
        if random_trajectory_length:

            # integration_steps_fn = lambda k: 10
            # Halton sequence
            integration_steps_fn = lambda k: jnp.ceil(
                jax.random.uniform(k) * rescale(num_steps_per_traj)
            ).astype('int32')
        else:
            integration_steps_fn = lambda _: num_steps_per_traj.astype(jnp.int32)

        alg = blackjax.dynamic_hmc(
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            integration_steps_fn=integration_steps_fn,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            inverse_mass_matrix=inverse_mass_matrix,
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
            progress_bar=True,
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