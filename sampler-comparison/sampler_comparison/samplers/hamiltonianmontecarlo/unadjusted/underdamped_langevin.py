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
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState, make_L_step_size_adaptation



def unadjusted_lmc_no_tuning(
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
    desired_energy_var=5e-4,
    desired_energy_var_max_ratio=jnp.inf,
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

        alg = blackjax.langevin(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
            desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        )

        if return_samples:
            transform = lambda state, info: (
                model.default_event_space_bijector(state.position),
                info,
            )

            get_final_sample = lambda state, info: (model.default_event_space_bijector(state.position), info)

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

        # jax.debug.print("EEVPD {x}", x=jnp.var(info.energy_change)/model.ndims)

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


def unadjusted_lmc_tuning(
    initial_position,
    num_steps,
    rng_key,
    logdensity_fn,
    integrator_type,
    diagonal_preconditioning,
    num_tuning_steps=500,
    stage3=True,
    desired_energy_var=5e-4,
    desired_energy_var_max_ratio=jnp.inf,
    num_windows=2,
    params=None,
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

    tune_key, init_key, nuts_key = jax.random.split(rng_key, 3)

    frac_tune1 = num_tuning_steps / (3 * num_steps)
    frac_tune2 = num_tuning_steps / (3 * num_steps)
    frac_tune3 = num_tuning_steps / (3 * num_steps) if stage3 else 0.0

    initial_state = blackjax.langevin.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
        metric=blackjax.mcmc.metrics.default_metric(jnp.ones(initial_position.shape[0]))
    )

    kernel = lambda inverse_mass_matrix: blackjax.langevin.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        inverse_mass_matrix=inverse_mass_matrix,
        desired_energy_var=desired_energy_var,
        desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        # (1/desired_energy_var)*10000,
    )

    do_nuts_warmup=True
    if do_nuts_warmup:

        

        warmup = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, 
                    target_acceptance_rate = 0.8, 
                    integrator=map_integrator_type_to_integrator["hmc"]['velocity_verlet'], 
                )
        
        (_, nuts_params), inf = warmup.run(nuts_key, initial_position, num_tuning_steps)

        inverse_mass_matrix = nuts_params['inverse_mass_matrix']
    
    else:
        inverse_mass_matrix = jnp.ones((initial_position.shape[0],))

    if params is None:

        params = MCLMCAdaptationState(
            1., 0.25 , inverse_mass_matrix=inverse_mass_matrix
        )
        # jax.debug.print("params {x}", x=params)

    # jax.debug.print("frac_tune1 {x}", x=frac_tune1)
    return blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=diagonal_preconditioning,
        frac_tune1=2.0,
        frac_tune2=frac_tune2,
        frac_tune3=frac_tune3,
        # frac_tune2=0.0,
        # frac_tune3=0.0,
        params=params,
        desired_energy_var=desired_energy_var,
        num_windows=num_windows,
        euclidean=True,
    )

    # jax.debug.print("frac_tune1 {x}", x=frac_tune1)

    # (blackjax_state_after_tuning, params) = make_L_step_size_adaptation(
    #                         kernel=kernel,
    #                         dim=initial_position.shape[0],
    #                         frac_tune1=1.0,
    #                         frac_tune2=0.0,
    #                         # target=0.9,
    #                         diagonal_preconditioning=False,
    #                         euclidean=True,
    #                         desired_energy_var=1e-1,
    #                     )(initial_state, params, num_steps, tune_key)
    
    # return (blackjax_state_after_tuning, params, jnp.inf)


def unadjusted_lmc(
    diagonal_preconditioning=True,
    integrator_type="velocity_verlet",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=3e-4,
    desired_energy_var_max_ratio=jnp.inf,
    return_only_final=False,
    num_windows=1,
    params=None,
    stage3=True,
    incremental_value_transform=None,
):
    def s(model, num_steps, initial_position, key):

        # logdensity_fn = lambda x : model.unnormalized_log_prob(make_transform(model)(x))

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_integrator_steps,
        ) = unadjusted_lmc_tuning(
            initial_position=initial_position,
            num_steps=num_steps,
            rng_key=tune_key,
            logdensity_fn=logdensity_fn,
            integrator_type=integrator_type,
            diagonal_preconditioning=diagonal_preconditioning,
            num_tuning_steps=num_tuning_steps,
            desired_energy_var=desired_energy_var,
            desired_energy_var_max_ratio=desired_energy_var_max_ratio,
            num_windows=num_windows,
            params=params,
            stage3=stage3,
        
        )
        # jax.debug.print("params {x}", x=(blackjax_mclmc_sampler_params))

        expectations, metadata = unadjusted_lmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mclmc_sampler_params.step_size,
            L=blackjax_mclmc_sampler_params.L,
            inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
            return_samples=return_samples,
            return_only_final=return_only_final,
            desired_energy_var=desired_energy_var,
            desired_energy_var_max_ratio=desired_energy_var_max_ratio,
            incremental_value_transform=incremental_value_transform,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s
