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
from src.nuts_tuning import da_adaptation
from src.samplers.general import with_only_statistics
from src.util import *






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
            model,
            alg,
            state,
            fast_key,
            num_steps,
            incremental_value_transform=incremental_value_transform,
            )
            expectations, info = results[0], results[1]

        # ess_corr = jax.lax.cond(
        #     not return_ess_corr,
        #     lambda: jnp.inf,
        #     lambda: jnp.mean(
        #         effective_sample_size(
        #             jax.vmap(lambda x: ravel_pytree(x)[0])(
        #                 run_inference_algorithm(
        #                     rng_key=slow_key,
        #                     initial_state=state,
        #                     inference_algorithm=alg,
        #                     num_steps=num_steps,
        #                     transform=lambda state, _: (
        #                         model.sample_transformations["square"](state.position)
        #                     ),  # TODO: transform?
        #                     progress_bar=False,
        #                 )[1]
        #             )[None, ...]
        #         )
        #     )
        #     / num_steps,
        # )
            
        

        return (
            expectations,
            {
                "params": params,
                "num_grads_per_proposal": info.num_integration_steps.mean()
                * calls_per_integrator_step(integrator_type),
                "acc_rate": info.acceptance_rate.mean(),
                "ess_corr": 0.0,
                "num_tuning_steps": num_tuning_steps,
            },
        )

    return s



