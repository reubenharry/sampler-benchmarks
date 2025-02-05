import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import sys

sys.path.append("../sampler-comparison")
from sampler_comparison.util import map_integrator_type_to_integrator
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc_no_tuning,
    adjusted_mclmc_tuning,
)
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_make_L_step_size_adaptation,
)
from blackjax.adaptation.mclmc_adaptation import make_L_step_size_adaptation
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from sampler_comparison.samplers.general import sampler_grads_to_low_error

def grid_search_only_L(
    model,
    sampler,
    num_steps,
    num_chains,
    integrator_type,
    key,
    grid_size,
    opt="max",
    grid_iterations=2,
):
    

    da_key, bench_key, init_pos_key, fast_tune_key = jax.random.split(key, 4)

    initial_position = jax.random.normal(
        shape=(
            model.ndims,
        ),
        key=init_pos_key,
    )
    jax.debug.print("log dens 2 {x}", x=(model.unnormalized_log_prob(initial_position)))
    # initial_positon = model.sample_init(init_pos_key)

    if sampler == "adjusted_mchmc":

        integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

        random_trajectory_length = True
        if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
                jax.random.uniform(k) * rescale(avg_num_integration_steps)
            )
        else:
            integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
                avg_num_integration_steps
            )

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        inverse_mass_matrix=inverse_mass_matrix,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=model.unnormalized_log_prob,
        L_proposal_factor=jnp.inf,
    )

        target_acc_rate = 0.9

        


        (blackjax_state_after_tuning, blackjax_sampler_params, _) = adjusted_mclmc_tuning(
            initial_position=initial_position,
            num_steps=num_steps,
            rng_key=fast_tune_key,
            logdensity_fn=model.unnormalized_log_prob,
            diagonal_preconditioning=False,
            target_acc_rate=target_acc_rate,
            random_trajectory_length=True,
            integrator=integrator,
            L_proposal_factor=jnp.inf,

            # kernel,
            # frac_tune3=0.0,
            params=None,
            max="avg",
            num_windows=2,
            tuning_factor=1.3,
            num_tuning_steps=1000,
        )

    # elif sampler=='adjusted_mclmc':

    #     integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    #     random_trajectory_length = False
    #     if random_trajectory_length:
    #         integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
    #         jax.random.uniform(k) * rescale(avg_num_integration_steps))
    #     else:
    #         integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

    #     kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc.build_kernel(
    #     integrator=integrator,
    #     integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
    #     inverse_mass_matrix=inverse_mass_matrix,
    #     )(
    #         rng_key=rng_key,
    #         state=state,
    #         step_size=step_size,
    #         logdensity_fn=model.logdensity_fn,
    #         L_proposal_factor=L_proposal_factor,
    #     )

    #     target_acc_rate = 0.9

    #     (
    #         blackjax_state_after_tuning,
    #         blackjax_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, fast_tune_key, model.logdensity_fn, False, target_acc_rate, kernel, frac_tune3=0.0, params=None, max='avg', num_windows=2, tuning_factor=1.3, num_tuning_steps=5000)

    # elif sampler=='mclmc':

    #     (blackjax_state_after_tuning, blackjax_sampler_params) = unadjusted_mclmc_tuning(
    #                 initial_position=initial_position,
    #                 num_steps=num_steps,
    #                 rng_key=fast_tune_key,
    #                 logdensity_fn=model.logdensity_fn,
    #                 integrator_type=integrator_type,
    #                 diagonal_preconditioning=False,
    #                 num_windows=2,
    #                 num_tuning_steps=5000
    #             )

    # elif sampler=='adjusted_hmc':

    #     integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

    #     random_trajectory_length = True
    #     if random_trajectory_length:
    #         integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
    #         jax.random.uniform(k) * rescale(avg_num_integration_steps)).astype(jnp.int32)
    #     else:
    #         integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps).astype(jnp.int32)

    #     kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.dynamic_hmc.build_kernel(
    #     integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
    #     integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
    #     )(
    #         rng_key=rng_key,
    #         state=state,
    #         logdensity_fn=model.logdensity_fn,
    #         step_size=step_size,
    #         inverse_mass_matrix=jnp.diag(jnp.ones(model.ndims)),
    #     )

    #     (
    #         blackjax_state_after_tuning,
    #         blackjax_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, rng_key=fast_tune_key, logdensity_fn=model.logdensity_fn,diagonal_preconditioning=False, target_acc_rate=0.9, kernel=kernel, frac_tune3=0.1,  params=None, max='avg', num_windows=2,tuning_factor=1.3, num_tuning_steps=5000)

    # else:
    #     raise Exception(f"sampler {sampler} not recognized")

    z = blackjax_sampler_params.L
    state = blackjax_state_after_tuning
    jax.debug.print("initial L {x}", x=(z))

    # Lgrid = np.array([z])
    ESS = jnp.zeros(grid_size)
    ESS_AVG = jnp.zeros(grid_size)
    ESS_CORR_AVG = jnp.zeros(grid_size)
    ESS_CORR_MAX = jnp.zeros(grid_size)
    STEP_SIZE = jnp.zeros(grid_size)
    RATE = jnp.zeros(grid_size)
    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    for grid_iteration in range(grid_iterations):

        da_key_per_iter = jax.random.fold_in(da_key, grid_iteration)
        bench_key_per_iter = jax.random.fold_in(bench_key, grid_iteration)

        if grid_iteration == 0:
            Lgrid = jnp.linspace(z / 3, z * 3, grid_size)
        else:

            Lgrid = jnp.linspace(Lgrid[iopt - 1], Lgrid[iopt + 1], grid_size)
        jax.debug.print("Lgrid {x}", x=(Lgrid))

        for i in range(len(Lgrid)):
            da_key_per_iter = jax.random.fold_in(da_key_per_iter, i)
            bench_key_per_iter = jax.random.fold_in(bench_key_per_iter, i)
            jax.debug.print("L {x}", x=(Lgrid[i]))
            print("i", i)

            params = MCLMCAdaptationState(
                L=Lgrid[i],
                step_size=Lgrid[i] / 5,
                inverse_mass_matrix=1.0,
            )

            if sampler in ["adjusted_mclmc", "adjusted_mchmc", "adjusted_hmc"]:

                (blackjax_state_after_tuning, params, _, _) = (
                    adjusted_mclmc_make_L_step_size_adaptation(
                        kernel=kernel,
                        dim=model.ndims,
                        frac_tune1=0.1,
                        frac_tune2=0.0,
                        target=0.9,
                        diagonal_preconditioning=False,
                        fix_L_first_da=True,
                    )(state, params, num_steps, da_key_per_iter)
                )

            # elif sampler=='mclmc':

            #     kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            #         logdensity_fn=model.logdensity_fn,
            #         integrator=integrator,
            #         inverse_mass_matrix=inverse_mass_matrix,
            #     )

            #     (
            #         blackjax_state_after_tuning,
            #         params,
            #     ) = make_L_step_size_adaptation(
            #         kernel=kernel,
            #         dim=model.ndims,
            #         frac_tune1=0.1,
            #         frac_tune2=0.0,
            #         diagonal_preconditioning=False,
            #     )(
            #         state, params, num_steps, da_key_per_iter
            #     )

            # elif sampler=='adjusted_hmc':

            # jax.debug.print("DA {x}", x=(final_da))
            jax.debug.print(
                "benchmarking with L and step size {x}", x=(Lgrid[i], params.step_size)
            )

            if sampler == "adjusted_mchmc":

                (stats, sq_error) = sampler_grads_to_low_error(
                    model=model,
                    sampler=adjusted_mclmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=blackjax_state_after_tuning,
                        inverse_mass_matrix=1.0,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        L_proposal_factor=jnp.inf,
                        random_trajectory_length=True,
                        # return_ess_corr=False,
                    ),
                    key=bench_key_per_iter,
                    num_steps=num_steps,
                    batch_size=num_chains,
                )

            # elif sampler=='adjusted_mclmc':

            #     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            #         model,
            #         adjusted_mclmc_no_tuning(
            #             integrator_type=integrator_type,
            #             initial_state=blackjax_state_after_tuning,
            #             inverse_mass_matrix=1.0,
            #             L=Lgrid[i],
            #             step_size=params.step_size,
            #             L_proposal_factor=L_proposal_factor,
            #             random_trajectory_length=False,
            #             return_ess_corr=False,
            #             num_tuning_steps=0,
            #         ),
            #         bench_key_per_iter,
            #         n=num_steps,
            #         batch=num_chains,
            #     )

            # elif sampler=='adjusted_hmc':

            #     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            #         model,
            #         adjusted_hmc_no_tuning(
            #             integrator_type=integrator_type,
            #             initial_state=blackjax_state_after_tuning,
            #             inverse_mass_matrix=1.0,
            #             L=Lgrid[i],
            #             step_size=params.step_size,
            #             return_ess_corr=False,
            #             num_tuning_steps=0,
            #         ),
            #         bench_key_per_iter,
            #         n=num_steps,
            #         batch=num_chains,
            #     )

            # elif sampler=='mclmc':

            #     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
            #         model,
            #         unadjusted_mclmc_no_tuning(
            #             integrator_type=integrator_type,
            #             initial_state=blackjax_state_after_tuning,
            #             inverse_mass_matrix=1.0,
            #             L=Lgrid[i],
            #             step_size=params.step_size,
            #             return_ess_corr=False,
            #             num_tuning_steps=0,
            #         ),
            #         bench_key_per_iter,
            #         n=num_steps,
            #         batch=num_chains,
            #     )

            # else:
            #     raise Exception("sampler not recognized")

            # jax.debug.print("{x} ess", x=(ess))
            ESS = ESS.at[i].set(stats['max_over_parameters']['square']['grads_to_low_error'])
            ESS_AVG = ESS_AVG.at[i].set(stats['avg_over_parameters']['square']['grads_to_low_error'])
            # ESS_CORR_AVG[i] = ess_corr.mean().item()
            # STEP_SIZE[i] = params.step_size.mean().item()
            STEP_SIZE = STEP_SIZE.at[i].set(stats['step_size'])
            # RATE[i] = acceptance_rate.mean().item()
            # RATE = RATE.at[i].set(acceptance_rate.mean().item())
        # iopt = np.argmax(ESS)
        if opt == "max":
            iopt = np.argmax(ESS)
        elif opt == "avg":
            iopt = np.argmax(ESS_AVG)
        else:
            raise Exception("opt not recognized")
        edge = grid_iteration == 0 and (iopt == 0 or iopt == (len(Lgrid) - 1))

        print("iopt", iopt)
        jax.debug.print("optimal ess {x}", x=(ESS[iopt], ESS_AVG[iopt]))

    return Lgrid[iopt], STEP_SIZE[iopt], ESS[iopt], ESS_AVG[iopt], edge


from sampler_evaluation.models.banana import banana

model = banana()

print(model.unnormalized_log_prob(jnp.array([0.0, 0.0])))

L, step_size, ess, ess_avg, edge = grid_search_only_L(
    model=model,
    sampler="adjusted_mchmc",
    num_steps=1000,
    num_chains=2,
    integrator_type="velocity_verlet",
    key=jax.random.PRNGKey(0),
    grid_size=5,
    grid_iterations=2,
    opt="max",
)

print(L)
