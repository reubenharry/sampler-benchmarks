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
from blackjax.adaptation.adjusted_mclmc_adaptation import (
    adjusted_mclmc_make_L_step_size_adaptation,
)
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from sampler_comparison.samplers.general import sampler_grads_to_low_error
from sampler_comparison.samplers.general import (
    make_log_density_fn,
)


def grid_search_only_L(
    model,
    num_steps,
    num_chains,
    integrator_type,
    key,
    grid_size,
    opt="max",
    grid_iterations=2,
):

    logdensity_fn = make_log_density_fn(model)

    da_key, bench_key, init_pos_key, init_key, nuts_key = jax.random.split(key, 5)

    initial_position = jax.random.normal(
        shape=(model.ndims,),
        key=init_pos_key,
    )

    alg = blackjax.nuts

    warmup = blackjax.window_adaptation(
        alg,
        logdensity_fn,
        target_acceptance_rate=0.8,
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
    )

    (blackjax_state_after_tuning, nuts_params), inf = warmup.run(
        nuts_key, initial_position, 10000
    )
    blackjax_state_after_tuning = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=blackjax_state_after_tuning.position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )

    inverse_mass_matrix = nuts_params["inverse_mass_matrix"]

    # jax.debug.print("initial inverse mass matrix {x}", x=(inverse_mass_matrix))

    # initial_positon = model.sample_init(init_pos_key)

    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps)
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        inverse_mass_matrix=inverse_mass_matrix,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        L_proposal_factor=jnp.inf,
    )

    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    z = jnp.sqrt(model.ndims)
    state = blackjax_state_after_tuning
    jax.debug.print("initial L {x}", x=(z))

    # Lgrid = np.array([z])
    Num_Grads = jnp.zeros(grid_size)
    Num_Grads_AVG = jnp.zeros(grid_size)
    STEP_SIZE = jnp.zeros(grid_size)

    for grid_iteration in range(grid_iterations):

        da_key_per_iter = jax.random.fold_in(da_key, grid_iteration)
        bench_key_per_iter = jax.random.fold_in(bench_key, grid_iteration)

        if grid_iteration == 0:
            Lgrid = jnp.linspace(z / 5, z * 5, grid_size)
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
                inverse_mass_matrix=inverse_mass_matrix,
            )

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

            (stats, sq_error) = sampler_grads_to_low_error(
                model=model,
                sampler=adjusted_mclmc_no_tuning(
                    initial_state=blackjax_state_after_tuning,
                    integrator_type=integrator_type,
                    inverse_mass_matrix=inverse_mass_matrix,
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

            Num_Grads = Num_Grads.at[i].set(
                stats["max_over_parameters"]["square"]["grads_to_low_error"]
            )
            jax.debug.print(
                "benchmarking with L and step size {x}",
                x=(Lgrid[i], params.step_size, Num_Grads),
            )
            Num_Grads_AVG = Num_Grads_AVG.at[i].set(
                stats["avg_over_parameters"]["square"]["grads_to_low_error"]
            )
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
