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


def unadjusted_mclmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
):
    """
    Args:
        initial_state: Initial state of the chain
        integrator_type: Type of integrator to use (e.g. velocity verlet, mclachlan...)
        step_size: Step size to use
        L: Number of steps to run the chain for
        inverse_mass_matrix: Inverse mass matrix to use
        return_samples: Whether to return the samples or not
    Returns:
        A tuple of the form (expectations, stats) where expectations are the expectations of the chain and stats are the hyperparameters of the chain (L, stepsize and inverse mass matrix) and other metadata
    """

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        fast_key, slow_key = jax.random.split(key, 2)

        alg = blackjax.mclmc(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
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
                model=model,
                alg=alg,
                initial_state=initial_state,
                rng_key=fast_key,
                num_steps=num_steps,
            )
            expectations, info = results[0], results[1]

        return (
            expectations,
            {
                "L": L,
                "step_size": step_size,
                "acc_rate": jnp.nan,
                "num_tuning_grads": 0,
                "num_grads_per_proposal": calls_per_integrator_step(integrator_type),
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
    num_tuning_steps=500,
):
    """
    Args:
        initial_position: Initial position of the chain
        num_steps: Number of steps to run the chain for
        rng_key: Random number generator key
        logdensity_fn: Log density function of the target distribution
        integrator_type: Type of integrator to use (e.g. velocity verlet, mclachlan...)
        diagonal_preconditioning: Whether to use diagonal preconditioning
        num_tuning_steps: Number of tuning steps to use
    Returns:
        A tuple of the form (state, params) where state is the state of the chain after tuning and params are the hyperparameters of the chain (L, stepsize and inverse mass matrix)
    """

    tune_key, init_key = jax.random.split(rng_key, 2)

    frac_tune1 = num_tuning_steps / (3 * num_steps)
    frac_tune2 = num_tuning_steps / (3 * num_steps)
    frac_tune3 = num_tuning_steps / (3 * num_steps)

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
    )

    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        inverse_mass_matrix=inverse_mass_matrix,
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
    diagonal_preconditioning=True,
    integrator_type="mclachlan",
    num_tuning_steps=10000,
    return_samples=False,
):
    def s(model, num_steps, initial_position, key):

        # logdensity_fn = lambda x : model.unnormalized_log_prob(make_transform(model)(x))

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_integrator_steps,
        ) = unadjusted_mclmc_tuning(
            initial_position=initial_position,
            num_steps=num_steps,
            rng_key=tune_key,
            logdensity_fn=logdensity_fn,
            integrator_type=integrator_type,
            diagonal_preconditioning=diagonal_preconditioning,
            num_tuning_steps=num_tuning_steps,
        )

        expectations, metadata = unadjusted_mclmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mclmc_sampler_params.step_size,
            L=blackjax_mclmc_sampler_params.L,
            inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
            return_samples=return_samples,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s
