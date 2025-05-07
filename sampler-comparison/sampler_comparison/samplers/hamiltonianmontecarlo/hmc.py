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
    L_proposal_factor=jnp.inf,
):
    
    # print("Begin")

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

        # print(integration_steps_fn(jax.random.key(0)))

        alg = blackjax.dynamic_malt(
            logdensity_fn=logdensity_fn,
            step_size=step_size,
            integration_steps_fn=integration_steps_fn,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            inverse_mass_matrix=inverse_mass_matrix,
            L_proposal_factor=L_proposal_factor,
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

def adjusted_hmc_tuning(
    initial_position,
    num_steps,
    rng_key,
    logdensity_fn,
    diagonal_preconditioning,
    target_acc_rate,
    random_trajectory_length,
    integrator,
    # L_proposal_factor,
    params=None,
    max="avg",
    num_windows=1,
    tuning_factor=1.0,
    num_tuning_steps=500,
    L_factor_stage_3=0.3,
    warmup='nuts',
    stage_3=True,
    L_proposal_factor=jnp.inf,
):

    init_key, tune_key = jax.random.split(rng_key, 2)

    initial_state = blackjax.mcmc.dynamic_malt.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )


    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        ).astype('int32')
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
            avg_num_integration_steps
        ).astype('int32')


    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.dynamic_malt.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        L_proposal_factor=L_proposal_factor,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        # L_proposal_factor=L_proposal_factor,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    total_tuning_integrator_steps = 0

    tune_key_unadjusted, re_init_key, tune_key, tune_key_2, stage3_key = (
            jax.random.split(tune_key, 5)
        )

    if warmup=='unadjusted_mclmc':

        for i in range(num_windows):
            tune_key_unadjusted = jax.random.fold_in(tune_key_unadjusted, i)
            
            (
                blackjax_state_after_tuning,
                sampler_params,
                num_tuning_integrator_steps,
            ) = unadjusted_mclmc_tuning(
                initial_position=initial_position,
                num_steps=num_steps,
                rng_key=tune_key_unadjusted,
                logdensity_fn=logdensity_fn,
                integrator_type="velocity_verlet",
                diagonal_preconditioning=diagonal_preconditioning,
                num_tuning_steps=num_tuning_steps/num_windows,
                stage3=False,
            )
            # jax.debug.print("num_tuning_steps/num_windows {x}", x=num_tuning_integrator_steps)
            total_tuning_integrator_steps += num_tuning_integrator_steps

        dim = initial_state.position.shape[0]
        new_step_size = jnp.clip(sampler_params.step_size, max=sampler_params.L-0.01)
        sampler_params = sampler_params._replace(
            L=jnp.sqrt(dim),
            step_size=new_step_size,
        )

    elif warmup=='nuts':

        

        

        

        warmup = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, 
                    target_acceptance_rate = 0.8, 
                    integrator=map_integrator_type_to_integrator["hmc"]['velocity_verlet'], 
                )
        
        (blackjax_state_after_tuning, unadjusted_params), inf = warmup.run(tune_key_unadjusted, initial_position, num_tuning_steps)
        total_tuning_integrator_steps += inf.info.num_integration_steps.sum()

        dim = initial_state.position.shape[0]
        new_step_size = jnp.clip(unadjusted_params["step_size"], max=1.0-0.01)
        sampler_params = MCLMCAdaptationState(
            L=1.,
            step_size=new_step_size,
            inverse_mass_matrix=unadjusted_params['inverse_mass_matrix'],
        )
        # jax.debug.print("step size after nuts tuning {x}",x=new_step_size)
        # raise Exception("foo")


    state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=blackjax_state_after_tuning.position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=re_init_key,
    )

    # jax.debug.print("completed nuts stage {x}",x=state)


    (
        blackjax_state_after_tuning,
        sampler_params,
        _,
        num_tuning_integrator_steps,
    ) = adjusted_mclmc_make_L_step_size_adaptation(
        kernel=kernel,
        dim=dim,
        frac_tune1=2000 / num_steps,
        frac_tune2=0.0,
        target=target_acc_rate,
        diagonal_preconditioning=diagonal_preconditioning,
        max=max,
        tuning_factor=tuning_factor,
        fix_L_first_da=True,
        # euclidean=True,
    )(
        state, sampler_params, num_steps, tune_key
    )
    # jax.debug.print("step size after tuning {x}",x=sampler_params.step_size)
    total_tuning_integrator_steps += num_tuning_integrator_steps


    if stage_3:

        (
            blackjax_state_after_tuning,
            sampler_params,
            num_tuning_integrator_steps,
        ) = adjusted_mclmc_make_adaptation_L(
            kernel,
            frac=10000 / num_steps,
            Lfactor=L_factor_stage_3,
            max="avg",
            eigenvector=None,
        )(
            blackjax_state_after_tuning,
            sampler_params,
            num_steps,
            stage3_key,
        )

        total_tuning_integrator_steps += num_tuning_integrator_steps

        (
            blackjax_state_after_tuning,
            sampler_params,
            _,
            num_tuning_integrator_steps,
        ) = adjusted_mclmc_make_L_step_size_adaptation(
            kernel=kernel,
            dim=dim,
            frac_tune1=2000 / num_steps,
            frac_tune2=0.0,
            target=target_acc_rate,
            diagonal_preconditioning=diagonal_preconditioning,
            max=max,
            tuning_factor=tuning_factor,
            fix_L_first_da=True,
        )(
            state, sampler_params, num_steps, tune_key_2
        )
        total_tuning_integrator_steps += num_tuning_integrator_steps

    return (
        blackjax_state_after_tuning,
        sampler_params,
        total_tuning_integrator_steps,
    )


def adjusted_hmc(
    integrator_type="velocity_verlet",
    diagonal_preconditioning=True,
    # L_proposal_factor=jnp.inf,
    target_acc_rate=0.9,
    max="avg",
    num_windows=2,
    random_trajectory_length=True,
    tuning_factor=1.0,
    num_tuning_steps=20000,
    L_factor_stage_3=0.3,
    return_samples=False,
    return_only_final=False,
    stage_3=True,
    L_proposal_factor=jnp.inf,
    warmup='nuts',
):
    """
    Args:
    integrator_type: the integrator to use (e.g. 'mclachlan')
    diagonal_preconditioning: whether to use diagonal preconditioning
    L_proposal_factor: the factor to multiply L by to get the L value used in the proposal (infinite L means no Langevin noise in proposal)
    target_acc_rate: the target acceptance rate
    params: the initial parameters to use in the adaptation. If None, default parameters are used
    max: the method to use to calculate L in the first stage of adaptation for L
    num_windows: the number of windows to use in the adaptation
    random_trajectory_length: whether to use random trajectory length
    tuning_factor: the factor to multiply L by in the first stage of adaptation for L
    num_tuning_steps: the number of tuning steps to use
    L_factor_stage_3: the factor to multiply L by in the third stage of adaptation for L
    return_samples: whether to return samples
    warmup: whether to do a NUTS warmup or a warmup with MCLMC
    Returns:
    A function that runs the sampler
    """

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        (
            blackjax_state_after_tuning,
            sampler_params,
            num_tuning_integrator_steps,
        ) = adjusted_hmc_tuning(
            initial_position=initial_position,
            num_steps=num_steps,
            rng_key=tune_key,
            logdensity_fn=logdensity_fn,
            diagonal_preconditioning=diagonal_preconditioning,
            target_acc_rate=target_acc_rate,
            random_trajectory_length=random_trajectory_length,
            integrator=integrator,
            # L_proposal_factor=L_proposal_factor,
            # params=initial_params,
            max=max,
            num_windows=num_windows,
            tuning_factor=tuning_factor,
            num_tuning_steps=num_tuning_steps,
            L_factor_stage_3=L_factor_stage_3,
            warmup=warmup,
            stage_3=stage_3,
            L_proposal_factor=L_proposal_factor,
        )

        # jax.debug.print("initial state {x}",x=blackjax_state_after_tuning)
        # jax.debug.print("step size {x}",x=sampler_params.step_size)
        # raise Exception(sampler_params.step_size)

        expectations, metadata = adjusted_hmc_no_tuning(
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=sampler_params.step_size,
            L=sampler_params.L,
            inverse_mass_matrix=sampler_params.inverse_mass_matrix,
            # L_proposal_factor=L_proposal_factor,
            random_trajectory_length=random_trajectory_length,
            return_samples=return_samples,
            return_only_final=return_only_final,
            L_proposal_factor=L_proposal_factor,
        )(model, num_steps, initial_position, run_key)

        # jax.debug.print("intermediate {x}",x=expectations[0,:])
        # print("intermediate", expectations.shape)



        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s