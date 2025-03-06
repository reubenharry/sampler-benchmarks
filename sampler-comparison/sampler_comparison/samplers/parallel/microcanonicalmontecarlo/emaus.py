import functools
import itertools

import chex
import jax
# jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import optax
from absl.testing import absltest, parameterized

import blackjax
import blackjax.diagnostics as diagnostics
import blackjax.mcmc.random_walk
from blackjax.adaptation.base import get_filter_adapt_info_fn, return_all_adapt_info
from blackjax.adaptation.ensemble_mclmc import emaus
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.mcmc.integrators import isokinetic_mclachlan
from blackjax.util import run_inference_algorithm

from sampler_comparison.samplers.general import (
    make_log_density_fn,
)
from sampler_evaluation.evaluation.ess import samples_to_low_error


import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from blackjax.adaptation.ensemble_mclmc import emaus

# from sampler_evaluation.models.banana_mams_paper import banana_mams_paper




def parallel_microcanonical(num_steps1, num_steps2, num_chains, mesh, diagonal_preconditioning=True):

    def s(model, num_steps, initial_position, key):

        logdensity_fn = make_log_density_fn(model)

        def contract(e_x):
            bsq = jnp.square(e_x - model.sample_transformations["square"].ground_truth_mean) / (model.sample_transformations["square"].ground_truth_standard_deviation**2)
            return jnp.array([jnp.max(bsq), jnp.average(bsq)])
        
        observables_for_bias = lambda position: jnp.square(
            model.default_event_space_bijector(jax.flatten_util.ravel_pytree(position)[0])
        )

        info, grads_per_step, _acc_prob, final_state = emaus(
    
            logdensity_fn=logdensity_fn, 
            sample_init=model.exact_sample, 
            ndims=model.ndims, 
            num_steps1=num_steps1, 
            num_steps2=num_steps2, 
            num_chains=num_chains, 
            mesh=mesh, 
            rng_key=jax.random.key(0), 
            early_stop=True,
            diagonal_preconditioning=diagonal_preconditioning, 
            integrator_coefficients= None, 
            steps_per_sample=15,
            ensemble_observables= lambda x: x,
            observables_for_bias=observables_for_bias,
            contract = contract,
            ) 
        

        jax.debug.print("info {x}", x=info["phase_2"][0]['bias'])

        bias = info["phase_2"][0]["bias"]

        jax.debug.print("samples to low error {x}", x=samples_to_low_error(bias[:,0], low_error=0.01))
        jax.debug.print("samples to low error {x}", x=samples_to_low_error(bias[:,1], low_error=0.01))

        return final_state.position, {
                "L": info["phase_2"][0]["L"],
                "step_size": info["phase_2"][0]["step_size"],
                "num_grads_per_proposal": info["phase_2"][0]["steps_per_sample"],
                "acc_rate": info["phase_2"][0]["acc_prob"],
                "num_tuning_grads": jnp.array([jnp.nan]),
            },
    
    return s
