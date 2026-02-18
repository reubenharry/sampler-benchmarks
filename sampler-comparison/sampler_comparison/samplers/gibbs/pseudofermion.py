from functools import partial
import jax
import sys
import os
# print(os.listdir("../../blackjax"))
sys.path.append("../../blackjax")
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
# print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))
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

def unadjusted_mclmc_no_tuning_pseudofermion(
    model,
    initial_state,
    integrator_type,
    step_size,
    L,
    inverse_mass_matrix,
    return_samples=False,
    incremental_value_transform=None,
    return_only_final=False,
    
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

        # base_alg = blackjax.mclmc(
        #     logdensity_fn=logdensity_fn,
        #     L=L,
        #     step_size=step_size,
        #     inverse_mass_matrix=inverse_mass_matrix,
        #     integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
        # )

        alg = blackjax.pseudofermion(
            logdensity_fn=logdensity_fn,
            L=L,
            step_size=step_size,
            kernel_1=blackjax.mclmc.build_kernel(
                integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
                desired_energy_var_max_ratio=jnp.inf,
                desired_energy_var=5e-4,
            ),
            kernel_2=lambda key, x: model.Mpsi( sample_noise_complex(shape=model.ndims, rng_key=key), x.position),
            init_1=blackjax.mclmc.init,
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

def unadjusted_mclmc_pseudofermion(
    diagonal_preconditioning=True,
    integrator_type="mclachlan",
    num_tuning_steps=20000,
    return_samples=False,
    desired_energy_var=5e-4,
    return_only_final=False,
    incremental_value_transform=None,
    alba_factor=0.4,
    num_alba_steps=None,
):
    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        tune_key, run_key = jax.random.split(key, 2)
        
        warmup = unadjusted_alba(
            mcmc_kernel=blackjax.pseudofermion.build_kernel(
                kernel_1=blackjax.mclmc.build_kernel(map_integrator_type_to_integrator["mclmc"][integrator_type]),
                kernel_2=lambda key, x: model.Mpsi( sample_noise_complex(shape=model.ndims, rng_key=key), x.position),
                # logdensity_fn=logdensity_fn,
            ),
            init=partial(blackjax.pseudofermion.init, pseudofermion=jnp.zeros(initial_position.shape[0], dtype=jnp.complex128), init_1=blackjax.mclmc.init, init_2=lambda x: x),
            logdensity_fn=logdensity_fn, 
            target_eevpd=desired_energy_var, 
            v=1., 
            num_alba_steps=num_tuning_steps // 3 if num_alba_steps is None else num_alba_steps,
            preconditioning=diagonal_preconditioning,
            alba_factor=alba_factor,
            )
        
        # jax.debug.print("running warmup")
        (blackjax_state_after_tuning, blackjax_mclmc_sampler_params), adaptation_info = warmup.run(tune_key, initial_position, num_tuning_steps)


        # jax.debug.print("warmup finished with {x}", x=blackjax_mclmc_sampler_params['step_size'])
        # jax.debug.print("state: {x}", x=blackjax_state_after_tuning)


        num_tuning_integrator_steps = num_tuning_steps

        # blackjax_state_after_tuning = blackjax.pseudofermion.init(
        #     initial_position, 
        #     pseudofermion=jnp.zeros(2*16*16, dtype=jnp.complex128),
        #     logdensity_fn=partial(logdensity_fn, pf=jnp.zeros(2*16*16, dtype=jnp.complex128)),
        #     rng_key=tune_key,
        #     init_1=blackjax.mclmc.init,
        #     init_2=lambda x: x
        # )

        # raise Exception("stop")

        expectations, metadata = unadjusted_mclmc_no_tuning_pseudofermion(
            model=model,
            initial_state=blackjax_state_after_tuning,
            integrator_type=integrator_type,
            # step_size=0.1,
            # L=1.0,
            # inverse_mass_matrix=jnp.ones(model.ndims),
            step_size=blackjax_mclmc_sampler_params['step_size'],
            L=blackjax_mclmc_sampler_params['L'],
            inverse_mass_matrix=blackjax_mclmc_sampler_params['inverse_mass_matrix'],
            return_samples=return_samples,
            return_only_final=return_only_final,
            incremental_value_transform=incremental_value_transform,
        )(model, num_steps, initial_position, run_key)

        return expectations, metadata | {
            "num_tuning_grads": num_tuning_integrator_steps
            * calls_per_integrator_step(integrator_type)
        }

    return s




# def unadjusted_mclmc_pseudofermion(
#     diagonal_preconditioning=True,
#     integrator_type="mclachlan",
#     num_tuning_steps=20000,
#     return_samples=False,
#     desired_energy_var=5e-4,
#     return_only_final=False,
#     incremental_value_transform=None,
#     alba_factor=0.4,
#     num_alba_steps=None,
#     # integrator_type="mclachlan",
#     # diagonal_preconditioning=True,
#     # return_samples=False,
#     # incremental_value_transform=None,
#     # num_tuning_steps=5000,
#     # return_only_final=False,
#     # # target_acc_rate=0.8,
#     # # cos_angle_termination=0.,
#     # progress_bar=False,
#     # get_fermion_matrix_fn=None,
#     # sample_temporary_state_fn=None,
#     # kernel_main=None,
#     # init_main=None,
#     # initial_params=None
# ):
    
    

#     def s(model, num_steps, initial_position, key):


#         logdensity_fn = make_log_density_fn(model)
#         integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

#         rng_key, warmup_key, init_key = jax.random.split(key, 3)

#         if initial_params is None :
#             if not diagonal_preconditioning:
#                 state, params, adaptation_info = da_adaptation(
#                     rng_key=warmup_key,
#                     initial_position=initial_position,
#                     algorithm=blackjax.nuts,
#                     integrator=integrator,
#                     logdensity_fn=logdensity_fn(fermion_matrix),
#                     num_steps=num_tuning_steps,
#                     target_acceptance_rate=target_acc_rate,
#                     # cos_angle_termination=cos_angle_termination,
#                 )
#                 from blackjax.mcmc.integrators import IntegratorState
#                 state = IntegratorState(state.position, None, state.logdensity, state.logdensity_grad)

#             else:
#                 warmup = blackjax.window_adaptation(
#                     blackjax.nuts, logdensity_fn, integrator=integrator,
#                     #  cos_angle_termination=cos_angle_termination
#                 )
#                 (state, params), adaptation_info = warmup.run(
#                     warmup_key, initial_position, num_tuning_steps
#                 )
#                 state.momentum = None

#                 adaptation_info = adaptation_info.info
        
#         else:
#             state = init_main(initial_position, logdensity_fn=logdensity_fn(fermion_matrix), rng_key=init_key)
#             params = initial_params
#             print("params step size", params["step_size"])

#         state = GibbsState(
#             position=state.position,
#             momentum=state.momentum,
#             logdensity=state.logdensity,
#             logdensity_grad=state.logdensity_grad,
#             pseudofermion=None,
#         )

        

        

#         alg = blackjax.pseudofermion(
#             kernel_1=kernel_1,
#             init_1=init_1,
#             kernel_2=kernel_2,
#             init_2=init_2,
#             logdensity_fn=model.log_density_fn,
#             # step_size=params["step_size"],
#             # inverse_mass_matrix=params["inverse_mass_matrix"],
#             get_fermion_matrix_fn=get_fermion_matrix_fn,
#             sample_temporary_state_fn=sample_temporary_state_fn,
#             # num_integration_steps=params["num_integration_steps"],
#         )

#         if return_samples:
#             transform = lambda state, info: (
#                 # model.default_event_space_bijector(state.position),
#                 state.position,
#                 info,
#             )

#             get_final_sample = lambda state, info: (
#                 # model.default_event_space_bijector(state.position), 
#                 state.position, 
#                 info
#             )

#         else:
#             alg, init, transform = with_only_statistics(
#                 model=model,
#                 alg=alg,
#                 incremental_value_transform=incremental_value_transform,
#             )

#             state = init(state)

#             get_final_sample = lambda output, info: (output[1][1], info)

#         final_output, history = run_inference_algorithm(
#             rng_key=rng_key,
#             initial_state=state,
#             inference_algorithm=alg,
#             num_steps=num_steps,
#             transform=(lambda a, b: None) if return_only_final else transform,
#             progress_bar=progress_bar,
#         )

#         if return_only_final:

#             return get_final_sample(final_output, {})

#         (expectations, info) = history

#         return (
#             expectations,
#             {
#                 "L": params["step_size"] * info.num_integration_steps.mean(),
#                 "step_size": params["step_size"],
#                 "num_grads_per_proposal": info.num_integration_steps.mean()
#                 * calls_per_integrator_step(integrator_type),
#                 "acc_rate": info.acceptance_rate.mean(),
#                 # "num_tuning_grads": info.adaptation_info.num_integration_steps.sum()
#                 # * calls_per_integrator_step(integrator_type),
#             },
#         )

#     return s

# if __name__ == "__main__":
#     position = jnp.ones(10)
#     pseudofermion = jnp.ones(10)
#     # your code will be imported here

#     def logdensity_fn(x, pseudofermion):
#         return 1.0

#     # nuts = blackjax.nuts(logdensity_fn)
    

        
    
#     state = blackjax.pseudofermion.init(
#         position=position, 
#         logdensity_fn=logdensity_fn, 
#         pseudofermion=pseudofermion, 
#         init_1=blackjax.hmc.init, 
#         init_2=lambda x: x,
#         rng_key=jax.random.key(0)
#     )
#     # print(state)

#     new_state = blackjax.pseudofermion.build_kernel(
#         kernel_1=blackjax.hmc.build_kernel(), 
#         kernel_2=lambda x: x, 
#         logdensity_fn=logdensity_fn)(state, jax.random.key(0))
#     print(new_state)
# # if __name__ == "__main__":
# #     import jax.numpy as jnp
# #     from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
# #     from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import hmc
# #     from sampler_comparison.samplers.hamiltonianmontecarlo.mala import mala
# #     from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.hmc import uhmc
# #     from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted
# #     from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted

# #     sampler = pseudofermion(
# #         integrator_type="velocity_verlet",
# #         diagonal_preconditioning=True,
# #         return_samples=True,
# #         num_tuning_steps=50,
# #         target_acc_rate=0.8,
# #     )

# #     model = phi4(1024, 4.0, load_from_file=False)


# #     samples, metadata = sampler(model, 1000, jnp.ones(model.ndims), jax.random.key(0))
# #     print(samples)
# #     print(metadata)