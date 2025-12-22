from functools import partial
import jax
import sys
import os
print(os.listdir("../../blackjax"))
sys.path.append("../../blackjax")
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))
# from blackjax.util import run_inference_algorithm
# import blackjax
# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts_tuning import da_adaptation
# from sampler_comparison.samplers.general import (
#     with_only_statistics,
#     make_log_density_fn,
# )
# from sampler_comparison.util import *
# from blackjax.mcmc.pseudofermion import GibbsState
import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
    sampler_grads_to_low_error,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.unadjusted_alba import unadjusted_alba
from blackjax.adaptation.adjusted_alba import adjusted_alba
import time
import blackjax
import jax.numpy as jnp

import jax
import jax.numpy as jnp
import blackjax
from blackjax.util import run_inference_algorithm
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
    sampler_grads_to_low_error,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.unadjusted_alba import unadjusted_alba
import time
from blackjax.mcmc.pseudofermion import sample_noise_complex
from blackjax.mcmc.adjusted_mclmc_dynamic import make_random_trajectory_length_fn

def adjusted_hmc_no_tuning_pseudofermion(
    model,
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
    L_proposal_factor=jnp.inf,
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

        integration_steps_fn = make_random_trajectory_length_fn(True)


        alg = blackjax.pseudofermion(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            kernel_1=lambda rng_key, state, step_size, L, inverse_mass_matrix, logdensity_fn: blackjax.dynamic_malt.build_kernel(
                integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
                # desired_energy_var_max_ratio=jnp.inf,
                # desired_energy_var=5e-4,
                L_proposal_factor=L_proposal_factor,
            )(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, integration_steps_fn(L/step_size)),
            kernel_2=lambda key, x: model.Mpsi( sample_noise_complex(shape=model.ndims, rng_key=key), x.position),
            init_1=blackjax.dynamic_malt.init,
            # init_2=lambda x: x,
            inverse_mass_matrix=inverse_mass_matrix,
            
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

def adjusted_hmc_pseudofermion(
    diagonal_preconditioning=True,
    integrator_type="velocity_verlet",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=5e-4,
    target_acc_rate=0.9,
    return_only_final=False,
    incremental_value_transform=None,
    alba_factor=0.4,
    num_alba_steps=None,
    L_proposal_factor=jnp.inf,
):
    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)

        integration_steps_fn = make_random_trajectory_length_fn(True)
        
        warmup = adjusted_alba(
            unadjusted_mcmc_kernel=blackjax.pseudofermion.build_kernel(
                kernel_1=blackjax.mclmc.build_kernel(map_integrator_type_to_integrator["mclmc"][integrator_type]),
                kernel_2=lambda key, x: model.Mpsi( sample_noise_complex(shape=model.ndims, rng_key=key), x.position),
                # logdensity_fn=logdensity_fn,
            ),
            unadjusted_init=partial(blackjax.pseudofermion.init, pseudofermion=jnp.zeros(initial_position.shape[0], dtype=jnp.complex128), init_1=blackjax.mclmc.init, init_2=lambda x: x),
            adjusted_mcmc_kernel=blackjax.pseudofermion.build_kernel(
                kernel_1=blackjax.dynamic_malt.build_kernel(integrator=map_integrator_type_to_integrator["hmc"][integrator_type], L_proposal_factor=L_proposal_factor),
            kernel_2=lambda key, x: model.Mpsi( sample_noise_complex(shape=model.ndims, rng_key=key), x.position),
            ),
            adjusted_init=partial(blackjax.pseudofermion.init, pseudofermion=jnp.zeros(initial_position.shape[0], dtype=jnp.complex128), init_1=blackjax.dynamic_malt.init, init_2=lambda x: x),
            logdensity_fn=logdensity_fn, 
            target_eevpd=desired_energy_var, 
            v=jnp.sqrt(model.ndims), 
            num_alba_steps=num_tuning_steps // 3 if num_alba_steps is None else num_alba_steps,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            target_acceptance_rate=target_acc_rate,
            L_proposal_factor=L_proposal_factor,
            integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
            )

       
        
        (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)

        num_tuning_integrator_steps = num_tuning_steps

        # print(blackjax_state_after_tuning, "state after tuning")
        # blackjax_state_after_tuning = blackjax.pseudofermion.init(
        #     initial_position, 
        #     pseudofermion=jnp.zeros(model.ndims, dtype=jnp.complex128),
        #     logdensity_fn=partial(logdensity_fn, pf=jnp.zeros(model.ndims, dtype=jnp.complex128)),
        #     random_generator_arg=tune_key,
        #     init_1=blackjax.dynamic_malt.init,
        #     init_2=lambda x: x
        # )




        expectations, metadata = adjusted_hmc_no_tuning_pseudofermion(
            model=model,
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            step_size=blackjax_mclmc_sampler_params['step_size'],
            L=blackjax_mclmc_sampler_params['L'],
            inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
            # step_size=1.0,
            # L=1.0,
            # inverse_mass_matrix=jnp.ones(model.ndims),
            return_samples=return_samples,
            return_only_final=return_only_final,
            incremental_value_transform=incremental_value_transform,
            L_proposal_factor=L_proposal_factor,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s

