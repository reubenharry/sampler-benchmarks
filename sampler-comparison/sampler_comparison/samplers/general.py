import jax
import jax.numpy as jnp
from blackjax.util import run_inference_algorithm
from blackjax.util import store_only_expectation_values
import sys, os

sys.path.append('../sampler-evaluation/')
from sampler_comparison.util import *
from sampler_evaluation.evaluation.ess import samples_to_low_error
from sampler_evaluation.evaluation.ess import get_standardized_squared_error
# awaiting switch to full pytree support
# make_transform = lambda model : lambda pos : jax.tree.map(lambda z, b: b(z), pos, model.default_event_space_bijector)
from blackjax.diagnostics import effective_sample_size
import itertools
import numpy as np

def frobenius(estimated_cov, true_cov):

    inv_cov = jnp.linalg.inv(true_cov)
    residual = jnp.eye(true_cov.shape[0]) - inv_cov @ estimated_cov
    # jax.debug.print("residual {x}", x=jnp.average(jnp.diag(residual@residual)))
    return jnp.average(jnp.diag(residual@residual))

# produce a kernel that only stores the average values of the bias for E[x_2] and Var[x_2]

def bias(expectation, f, model):
    # print(expectation.shape, "shape")

    # assert expectation.shape==1, expectation.shape
    # jax.debug.print("expectation {x}", x=expectation.shape)
    if len(expectation.shape) == 1:
        avg_bias = jnp.average(jnp.square(
                        expectation - model.sample_transformations[f].ground_truth_mean
                    ) / (
                        model.sample_transformations[f].ground_truth_standard_deviation
                        ** 2))
        max_bias = jnp.max(jnp.square(
                        expectation - model.sample_transformations[f].ground_truth_mean
                    ) / (
                        model.sample_transformations[f].ground_truth_standard_deviation
                        ** 2))
        # jax.debug.print("avg bias {x}", x=avg_bias)
        # jax.debug.print("max bias {x}", x=max_bias)
        return {
            'avg' : avg_bias,
            'max' : max_bias,
        }
    elif len(expectation.shape) == 2:
        return {
            'avg' : frobenius(expectation, model.sample_transformations[f].ground_truth_mean),
            'max' : frobenius(expectation, model.sample_transformations[f].ground_truth_mean)
        }

def with_only_statistics(model, alg, incremental_value_transform=None):

    if incremental_value_transform is None:
        
        incremental_value_transform = lambda expectations: jax.tree.map_with_path(lambda path, expectation: 
            bias(expectation=expectation, f=path[0].key, model=model),
            expectations)
        
        
    print([model.sample_transformations[trans].ground_truth_mean for trans in model.sample_transformations]
        )       

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: {trans: model.sample_transformations[trans].fn(model.default_event_space_bijector(state.position)) for trans in model.sample_transformations
        },
        incremental_value_transform=incremental_value_transform,
    )
    

    return memory_efficient_sampling_alg, memory_efficient_sampling_alg.init, transform


# this follows the inference_gym tutorial: https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/notebooks/inference_gym_tutorial.ipynb
def initialize_model(model, key):

    z = jax.random.normal(key=key, shape=(model.ndims,))

    # awaiting switch to full pytree support
    #   def random_initialization(shape, dtype):
    #     return jax.tree.map(lambda d, s: jax.random.normal(key=key, shape=s, dtype=d), dtype, shape)

    #   z = jax.tree.map(lambda d, b, s: random_initialization(b.inverse_event_shape(s), d),
    #                         model.dtype, model.default_event_space_bijector, model.event_shape)

    #   x = jax.tree.map(lambda z, b: b(z), z, model.default_event_space_bijector)

    return z


def make_log_density_fn(model):

    # if not hasattr(model, "unnormalized_log_prob"):
    #     import pymc
    #     from pymc.sampling.jax import get_jaxified_logp

    #     log_density_fn = get_jaxified_logp(model)
    #     if type(model) == pymc.model.core.Model:
    #         return log_density_fn

    if hasattr(model, "log_density_fn"):
        return model.log_density_fn

    def log_density_fn(z):
        return model._unnormalized_log_prob(
            model.default_event_space_bijector(z)
        ) + model.default_event_space_bijector.forward_log_det_jacobian(
            z, event_ndims=1
        )

    return log_density_fn

# make_log_density_fn = lambda model: lambda z: (
#     model.unnormalized_log_prob(model.default_event_space_bijector(z))
#     + model.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1)
# )


def sampler_grads_to_low_error(
    sampler, model, batch_size, key,
    calculate_ess_corr=False,
):



    keys = jax.random.split(key, batch_size)

    # this key is deliberately fixed to the same value: we want the set of initial positions to be the same for different samplers
    init_keys = jax.random.split(jax.random.key(2), batch_size)

    if hasattr(model, "sample_init") and model.sample_init is not None:
        initial_position = jax.vmap(model.sample_init)(init_keys)
        # jax.debug.print("using sample init {x}",x=True)
    else:
        initial_position = jax.vmap(lambda key: initialize_model(model, key))(init_keys)

    # initial_position = jnp.ones((batch_size, model.ndims,))

    # jax.debug.print("sampler_grads_to_low_error {x}", x=key)
    samples, metadata = sampler(
        keys,
        initial_position,
    )

    individual_chain_statistics = False
    if individual_chain_statistics:

        runs = jax.pmap(samples_to_low_error)(samples[:,:,1])

        failures = jnp.argwhere(jnp.isinf(runs))
        successes = jnp.argwhere(1-jnp.isinf(runs))

        num_failures = failures.shape[0]
        percent_failures = num_failures / batch_size
        good_runs = runs[successes.squeeze()]

        mean_runs = jnp.mean(good_runs)

        # jax.debug.print("\npercent failures\n {x}", x=percent_failures)

        std = jnp.std(good_runs* grad_evals_per_step)

        mean_grads = mean_runs* grad_evals_per_step

        # jax.debug.print("\nresult\n {x}", x=(mean_grads-std, mean_grads , mean_grads+std))

    grad_evals_per_step = metadata["num_grads_per_proposal"].mean()

    if calculate_ess_corr:

        ess_correlation_max = jnp.min(effective_sample_size(samples) / (samples.shape[1] * batch_size * grad_evals_per_step))
        ess_correlation_avg = jnp.mean(effective_sample_size(samples) / (samples.shape[1] * batch_size * grad_evals_per_step))

        ess_correlation = {
            "max": ess_correlation_max,
            "avg": ess_correlation_avg,
        }

        squared_errors = {
            trans : {
                'avg' : get_standardized_squared_error(samples, model.sample_transformations[trans].fn,
            model.sample_transformations[trans].ground_truth_mean,
            model.sample_transformations[trans].ground_truth_standard_deviation**2,
            contract_fn=jnp.mean),
                'max' : get_standardized_squared_error(samples, model.sample_transformations[trans].fn,
            model.sample_transformations[trans].ground_truth_mean,
            model.sample_transformations[trans].ground_truth_standard_deviation**2,
            contract_fn=jnp.max)
            }

            for trans in model.sample_transformations
        }
        

    else:

        squared_errors = samples
        # jax.debug.print("\n\n\nsquared errors {x}\n\n\n", x=squared_errors)
        # raise Exception("stop")
        ess_correlation = {'max': jnp.nan,
             'avg': jnp.nan}

    def contract_fn(x):
        return np.nanmedian(x, axis=0)

    # err_ = contract_fn(squared_errors['square']['avg'])
    # b2 = jnp.mean(err_[-1]*(model.sample_transformations['square'].ground_truth_standard_deviation**2)/(model.sample_transformations['square'].ground_truth_mean**2))
    # jax.debug.print("final error is {x}", x=b2)

    def estimate_std(errs):

        resampled_errs = jax.random.choice(jax.random.key(10), errs, shape=(100, errs.shape[0]))

        grads_to_low_error_resampled = [samples_to_low_error(np.nanmedian(x, axis=0)) for x in resampled_errs]

        return np.nanstd(grads_to_low_error_resampled)


    # print(new_samples.shape, "new samples shape")
    
    # jax.debug.print("new samples {x}", x=errs)
    # # std of errs   
    # jax.debug.print("std {x}", x=jnp.std(errs))

    # jax.debug.print("\n\nsquared_errors\n\n {x}", x=contract_fn(np.array(squared_errors['square']['avg'])).shape)

    # jax.debug.print("squared_errors blah {x}", x=squared_errors['square']['avg'][0][-1])
    # jax.debug.print("squared_errors {x}", x=squared_errors['square']['max'][0][-1])


    return (
        {
            f"{max}_over_parameters": {
                expectation: {
                    # "error": contract_fn(np.array(squared_errors[expectation][max])),
                    "grads_to_low_error": (
                        samples_to_low_error(
                            contract_fn(np.array(squared_errors[expectation][max])),
                        )
                        * grad_evals_per_step
                    ).item(),
                    "grads_to_low_error_std": estimate_std(np.array(squared_errors[expectation][max])),
                    "autocorrelation": ess_correlation[max]
                }
                            
                for expectation in model.sample_transformations.keys()
                }
                
            

        for max in ["max", "avg"]
        }
            | {
                "num_tuning_grads": metadata["num_tuning_grads"].mean().item(),
                "L": metadata["L"].mean().item(),
                "step_size": metadata["step_size"].mean().item(),
            },
            
        squared_errors
    )
