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
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc_no_tuning,
)
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_make_L_step_size_adaptation,
)
from blackjax.adaptation.mclmc_adaptation import make_L_step_size_adaptation
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from sampler_comparison.samplers.general import sampler_grads_to_low_error
from sampler_comparison.samplers.general import (
    make_log_density_fn,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import (
    adjusted_hmc_no_tuning,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc_no_tuning
import blackjax.mcmc.metrics as metrics
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import unadjusted_hmc_no_tuning

def grid_search_step_size(state, params, num_steps, da_key_per_iter):
    return params


def grid_search_L(
    model,
    num_steps,
    num_chains,
    integrator_type,
    key,
    grid_size,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    sampler_type='adjusted_mclmc',
    euclidean=False,
    L_proposal_factor=jnp.inf,
    target_expectation='square',
    desired_energy_var=1e-4,
    diagonal_preconditioning=True, 
    acc_rate=0.9,
):
    

    logdensity_fn = make_log_density_fn(model)

    da_key, bench_key, init_pos_key, init_key, nuts_key = jax.random.split(key, 5)

    initial_position = jax.random.normal(
        shape=(model.ndims,),
        key=init_pos_key,
    )

    alg = blackjax.nuts

    if acc_rate is None:
        nuts_acc_rate = 0.8
    else:
        nuts_acc_rate = acc_rate

    warmup = blackjax.window_adaptation(
        alg,
        logdensity_fn,
        target_acceptance_rate=nuts_acc_rate,
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
    )

    (blackjax_state_after_tuning, nuts_params), inf = warmup.run(
        nuts_key, initial_position, num_tuning_steps
    )
    blackjax_state_after_tuning = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=blackjax_state_after_tuning.position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    if diagonal_preconditioning:
        inverse_mass_matrix = nuts_params["inverse_mass_matrix"]
    else:
        inverse_mass_matrix = jnp.ones((model.ndims,))

    # jax.debug.print("initial inverse mass matrix {x}", x=(inverse_mass_matrix))

    # initial_positon = model.sample_init(init_pos_key)

    # integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps)
    ).astype(jnp.int32)

    if sampler_type=='adjusted_mclmc':

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            L_proposal_factor=jnp.inf,
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
        )

    elif sampler_type=='unadjusted_mclmc':
        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.mcmc.mclmc.build_kernel(
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=inverse_mass_matrix,
            L=L,
            # L_proposal_factor=L_proposal_factor,
        )

    elif sampler_type=='adjusted_hmc':
        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.dynamic_malt.build_kernel(
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        L_proposal_factor=L_proposal_factor,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            # L_proposal_factor=L_proposal_factor,
            inverse_mass_matrix=inverse_mass_matrix,
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        )

    elif sampler_type=='unadjusted_lmc':
        

        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.langevin.build_kernel(
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
            # desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
    
    elif sampler_type=='unadjusted_hmc':
        

        kernel = lambda inverse_mass_matrix: lambda rng_key, state, L, step_size: blackjax.uhmc.build_kernel(
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            desired_energy_var=desired_energy_var,
            # desired_energy_var_max_ratio=desired_energy_var_max_ratio,
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )
    else:
        raise Exception('sampler not recognized')
        

    # integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    z = 1 if euclidean else jnp.sqrt(model.ndims)
    state = blackjax_state_after_tuning
    # z = blackjax_p
    jax.debug.print("initial L {x}", x=(z))

    # Lgrid = np.array([z])
    Num_Grads = jnp.zeros(grid_size)
    Num_Grads_AVG = jnp.zeros(grid_size)
    Num_Grads_COV = jnp.zeros(grid_size)
    STEP_SIZE = jnp.zeros(grid_size)

    if acc_rate is None:
        acc_rate = 0.9

    for grid_iteration in range(grid_iterations):

        da_key_per_iter = jax.random.fold_in(da_key, grid_iteration)
        bench_key_per_iter = jax.random.fold_in(bench_key, grid_iteration)

        if grid_iteration == 0:
            Lgrid = jnp.linspace(z / 3, z * 3, grid_size)
        else:

            Lgrid = jnp.linspace(Lgrid[max(iopt - 1, 0)], Lgrid[min(iopt + 1, Lgrid.shape[0]-1)], grid_size)
        jax.debug.print("Lgrid {x}", x=(Lgrid))

        for i in range(len(Lgrid)):
            da_key_per_iter = jax.random.fold_in(da_key_per_iter, i)
            bench_key_per_iter = jax.random.fold_in(bench_key_per_iter, i)
            jax.debug.print("L {x}", x=(Lgrid[i]))
            print("i", i)

            params = MCLMCAdaptationState(
                L=Lgrid[i],
                step_size=Lgrid[i] / 5,
                inverse_mass_matrix=inverse_mass_matrix,
            )

            if sampler_type=='adjusted_mclmc':

                (blackjax_state_after_tuning, params, _, _) = (
                    adjusted_mclmc_make_L_step_size_adaptation(
                        kernel=kernel,
                        dim=model.ndims,
                        frac_tune1=0.1,
                        frac_tune2=0.0,
                        target=acc_rate,
                        diagonal_preconditioning=False,
                        fix_L_first_da=True,
                    )(state, params, num_steps, da_key_per_iter)
                )

                

                sampler = adjusted_mclmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        L_proposal_factor=jnp.inf,
                        random_trajectory_length=True,
                        # return_ess_corr=False,
                    )

            elif sampler_type=='unadjusted_mclmc':
                
                mclmc_state = blackjax.mcmc.mclmc.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                )

                (blackjax_state_after_tuning, params) = (
                        make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            # target=0.9,
                            diagonal_preconditioning=False,
                        )(mclmc_state, params, num_steps, da_key_per_iter)
                    )

                sampler = unadjusted_mclmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        # L_proposal_factor=jnp.inf,
                        # random_trajectory_length=True,
                        # return_ess_corr=False,
                    )
                
            elif sampler_type=='adjusted_hmc':
                (blackjax_state_after_tuning, params, _, _) = (
                    adjusted_mclmc_make_L_step_size_adaptation(
                        kernel=kernel,
                        dim=model.ndims,
                        frac_tune1=0.1,
                        frac_tune2=0.0,
                        target=acc_rate,
                        diagonal_preconditioning=False,
                        fix_L_first_da=True,
                    )(state, params, num_steps, da_key_per_iter)
                )

                

                sampler = adjusted_hmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        # L_proposal_factor=jnp.inf,
                        random_trajectory_length=True,
                        # return_ess_corr=False,
                    )
                
            elif sampler_type=='unadjusted_lmc':
                lmc_state = blackjax.mcmc.underdamped_langevin.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                    # metric=metrics.default_metric(inverse_mass_matrix),
                )

                
                
                (blackjax_state_after_tuning, params) = make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            # target=0.9,
                            diagonal_preconditioning=False,
                            euclidean=True,
                            desired_energy_var=desired_energy_var,
                        )(lmc_state, params, num_steps, da_key_per_iter)
                
                # (
                #     blackjax_state_after_tuning,
                #     blackjax_mclmc_sampler_params,
                #     num_tuning_integrator_steps,
                # ) =  blackjax.mclmc_find_L_and_step_size(
                #     mclmc_kernel=kernel,
                #     num_steps=num_steps,
                #     state=lmc_state,
                #     rng_key=da_key_per_iter,
                #     diagonal_preconditioning=True,
                #     frac_tune3=0.1,
                #     frac_tune2=0.0,
                #     frac_tune1=0.0,
                #     params=params,
                #     desired_energy_var=1e-1,
                #     num_windows=1,
                #     euclidean=True,
                # )
                
                # jax.debug.print("out {x}", x=blackjax_mclmc_sampler_params.step_size)
                jax.debug.print("step size {x}", x=params.step_size)

                # if sampler_type=='unadjusted_lmc':
                #     key_per_step_size = jax.random.fold_in(bench_key_per_iter, i)

                #     # params = grid_search_step_size(lmc_state, params, num_steps, da_key_per_iter)  
                #     step_sizes = jnp.linspace(params.step_size / 3, params.step_size * 3, 10)
                #     jax.debug.print("step sizes {x}", x=step_sizes)
                #     Num_Grads_AVG_step_size = jnp.zeros(len(step_sizes))
                #     Num_Grads_MAX_step_size = jnp.zeros(len(step_sizes))
                    
                #     do_grid_search_step_size = True
                #     if do_grid_search_step_size:
                #         for j, step_size in enumerate(step_sizes):

                #             sampler = unadjusted_lmc_no_tuning(
                #                 initial_state=blackjax_state_after_tuning,
                #                 integrator_type=integrator_type,
                #                 inverse_mass_matrix=inverse_mass_matrix,
                #                 L=Lgrid[i],
                #                 step_size=step_size,
                #             )

                #             (stats, sq_error) = sampler_grads_to_low_error(
                #                 model=model,
                #                 sampler=jax.pmap(
                #             lambda key, pos: sampler(
                #                 model=model, num_steps=num_steps, initial_position=pos, key=key
                #                 )
                #             ),
                #                 key=key_per_step_size,
                #                 # num_steps=num_steps,
                #                 batch_size=num_chains,
                #                 )

                #             Num_Grads_AVG_step_size = Num_Grads_AVG_step_size.at[j].set(
                #                 stats["avg_over_parameters"][target_expectation]["grads_to_low_error"]
                #             )
                #             Num_Grads_MAX_step_size = Num_Grads_MAX_step_size.at[j].set(
                #                 stats["max_over_parameters"][target_expectation]["grads_to_low_error"]
                #             )

                #             jax.debug.print("Num_Grads_AVG_step_size {x}", x=(Num_Grads_AVG_step_size[j], step_size, Lgrid[i]))
                #             jax.debug.print("Num_Grads_MAX_step_size {x}", x=(Num_Grads_MAX_step_size[j], step_size, Lgrid[i]))
                #             # jax.debug.print("Num_Grads_COV_step_size {x}", x=(stats["max_over_parameters"]["covariance"]["grads_to_low_error"]))
                    
                #         if opt=="max":
                #             iopt_step_size = np.argmin(Num_Grads_MAX_step_size)
                #         elif opt=="avg":
                #             iopt_step_size = np.argmin(Num_Grads_AVG_step_size)
                #         else:
                #             raise Exception("opt not recognized")
                    
                #         step_size = step_sizes[iopt_step_size]
                #         params = params._replace(step_size=step_size)
                
                
                sampler = unadjusted_lmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                    )

            elif sampler_type=='unadjusted_hmc':

                hmc_state = blackjax.mcmc.uhmc.init(
                    position=state.position,
                    logdensity_fn=logdensity_fn,
                    random_generator_arg=jax.random.key(0),
                    # metric=metrics.default_metric(inverse_mass_matrix),
                )

                
                
                (blackjax_state_after_tuning, params) = make_L_step_size_adaptation(
                            kernel=kernel,
                            dim=model.ndims,
                            frac_tune1=0.1,
                            frac_tune2=0.0,
                            # target=0.9,
                            diagonal_preconditioning=False,
                            euclidean=True,
                            desired_energy_var=desired_energy_var,
                        )(hmc_state, params, num_steps, da_key_per_iter)
                
                sampler = unadjusted_hmc_no_tuning(
                        initial_state=blackjax_state_after_tuning,
                        integrator_type=integrator_type,
                        inverse_mass_matrix=inverse_mass_matrix,
                        L=Lgrid[i],
                        step_size=params.step_size,
                    )
                
                
                
            

            (stats, sq_error) = sampler_grads_to_low_error(
                model=model,
                sampler=jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps, initial_position=pos, key=key
                )
            ),
                key=bench_key_per_iter,
                # num_steps=num_steps,
                batch_size=num_chains,
            )

            jax.debug.print("max and avg grads {x}", x=(
                stats["max_over_parameters"][target_expectation]["grads_to_low_error"],
                stats["avg_over_parameters"][target_expectation]["grads_to_low_error"],
                )
            )

            Num_Grads = Num_Grads.at[i].set(
                stats["max_over_parameters"][target_expectation]["grads_to_low_error"]
            )
            jax.debug.print(
                "benchmarking with L and step size {x}",
                x=(Lgrid[i], params.step_size, Num_Grads),
            )
            Num_Grads_AVG = Num_Grads_AVG.at[i].set(
                stats["avg_over_parameters"][target_expectation]["grads_to_low_error"]
            )
            jax.debug.print("num grads avg {x}", x=stats["avg_over_parameters"][target_expectation]["grads_to_low_error"])
            STEP_SIZE = STEP_SIZE.at[i].set(stats["step_size"])
        if opt == "max":
            iopt = np.argmin(Num_Grads)
        elif opt == "avg":
            iopt = np.argmin(Num_Grads_AVG)
        else:
            raise Exception("opt not recognized")
        edge = grid_iteration == 0 and (iopt == 0 or iopt == (len(Lgrid) - 1))

        print("iopt", iopt)
        jax.debug.print(
            "optimal Num_Grads {x}", x=(Num_Grads[iopt], Num_Grads_AVG[iopt])
        )

    return (
        Lgrid[iopt],
        STEP_SIZE[iopt],
        Num_Grads[iopt],
        Num_Grads_AVG[iopt],
        edge,
        inverse_mass_matrix,
        blackjax_state_after_tuning,
    )

def grid_search_adjusted_mclmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    target_expectation='square',
    diagonal_preconditioning=True,
    acc_rate=0.99,
):
    
    def s(model, num_steps, initial_position, key):

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=num_steps,
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='adjusted_mclmc',
            target_expectation=target_expectation,
            diagonal_preconditioning=diagonal_preconditioning,
            acc_rate=acc_rate,
        )

        sampler=adjusted_mclmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    L_proposal_factor=jnp.inf,
                    random_trajectory_length=True,
                    # return_ess_corr=False,
                    return_samples=return_samples,
                )
        
 

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


def grid_search_hmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    L_proposal_factor=jnp.inf,
    diagonal_preconditioning=True,
):
    
    def s(model, num_steps, initial_position, key):

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=num_steps,
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='adjusted_hmc',
            euclidean=True,
            L_proposal_factor=L_proposal_factor,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        sampler=adjusted_hmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    # L_proposal_factor=jnp.inf,
                    random_trajectory_length=True,
                    # return_ess_corr=False,
                    return_samples=return_samples,
                    L_proposal_factor=L_proposal_factor,
                )
        
        print("shapes", initial_position.shape)
        print("key", key.shape)

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s

def grid_search_unadjusted_mclmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    diagonal_preconditioning=True,
):
    
    def s(model, num_steps, initial_position, key):

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=num_steps,
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='unadjusted_mclmc',
            diagonal_preconditioning=diagonal_preconditioning,
        )

        sampler=unadjusted_mclmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    return_samples=return_samples,
                    # return_ess_corr=False,
                )
        
 
        

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s

def grid_search_unadjusted_lmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=1e-4,
    diagonal_preconditioning=True,
):
    
    def s(model, num_steps, initial_position, key):

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=num_steps,
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='unadjusted_lmc',
            euclidean=True,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        sampler=unadjusted_lmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    return_samples=return_samples,
                    # return_ess_corr=False,
                )
        
 
        

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


def grid_search_unadjusted_hmc(
    num_chains,
    integrator_type,
    grid_size=10,
    opt="max",
    grid_iterations=2,
    num_tuning_steps=10000,
    return_samples=False,
    desired_energy_var=1e-4,
    diagonal_preconditioning=True,
):
    
    def s(model, num_steps, initial_position, key):

        (
            L,
            step_size,
            num_grads,
            num_grads_avg,
            edge,
            inverse_mass_matrix,
            blackjax_state_after_tuning,
        ) = grid_search_L(
            model=model,
            num_steps=num_steps,
            num_chains=num_chains,
            integrator_type=integrator_type,
            key=jax.random.key(0),
            grid_size=grid_size,
            opt=opt,
            grid_iterations=grid_iterations,
            num_tuning_steps=num_tuning_steps,
            sampler_type='unadjusted_hmc',
            euclidean=True,
            desired_energy_var=desired_energy_var,
            diagonal_preconditioning=diagonal_preconditioning,
        )

        sampler=unadjusted_hmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
                    L=L,
                    step_size=step_size,
                    return_samples=return_samples,
                    # return_ess_corr=False,
                )
        
 
        

        return jax.pmap(
            lambda key, pos: sampler(
                model=model, num_steps=num_steps*4, initial_position=pos, key=key
                )
            )(key, initial_position)
        
    return s


