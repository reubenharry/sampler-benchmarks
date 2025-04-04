import jax
import jax.numpy as jnp
from blackjax.util import run_inference_algorithm
from blackjax.util import store_only_expectation_values

from sampler_comparison.util import *
from sampler_evaluation.evaluation.ess import samples_to_low_error
from sampler_evaluation.evaluation.ess import get_standardized_squared_error
# awaiting switch to full pytree support
# make_transform = lambda model : lambda pos : jax.tree.map(lambda z, b: b(z), pos, model.default_event_space_bijector)
from blackjax.diagnostics import effective_sample_size


# produce a kernel that only stores the average values of the bias for E[x_2] and Var[x_2]
def with_only_statistics(model, alg, incremental_value_transform=None):

    if incremental_value_transform is None:
        incremental_value_transform = lambda x: jnp.array(
            [
                jnp.average(
                    jnp.square(
                        x[1] - model.sample_transformations["square"].ground_truth_mean
                    )
                    / (
                        model.sample_transformations[
                            "square"
                        ].ground_truth_standard_deviation
                        ** 2
                    )
                ),
                jnp.max(
                    jnp.square(
                        x[1] - model.sample_transformations["square"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "square"
                    ].ground_truth_standard_deviation
                    ** 2
                ),
                jnp.average(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / (
                        model.sample_transformations[
                            "identity"
                        ].ground_truth_standard_deviation
                        ** 2
                    )
                ),
                jnp.max(
                    jnp.square(
                        x[0]
                        - model.sample_transformations["identity"].ground_truth_mean
                    )
                    / model.sample_transformations[
                        "identity"
                    ].ground_truth_standard_deviation
                    ** 2
                ),
            ]
        )

    outer_transform = model.sample_transformations["identity"] if callable(model.sample_transformations["identity"]) else lambda x:x

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=alg,
        state_transform=lambda state: jnp.array(
            [
                # model.sample_transformations["identity"](state.position),
                # model.sample_transformations["square"](state.position),
                outer_transform(
                    model.default_event_space_bijector(state.position)
                ),
                outer_transform(
                    model.default_event_space_bijector(state.position)
                )
                ** 2,
                outer_transform(
                    model.default_event_space_bijector(state.position)
                )
                ** 4,
            ]
        ),
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
        return model.unnormalized_log_prob(
            model.default_event_space_bijector(z)
        ) + model.default_event_space_bijector.forward_log_det_jacobian(
            z, event_ndims=1
        )

    return log_density_fn

    # return lambda z: model.unnormalized_log_prob(model.default_event_space_bijector(z))


# make_log_density_fn = lambda model: lambda z: (
#     model.unnormalized_log_prob(model.default_event_space_bijector(z))
#     + model.default_event_space_bijector.forward_log_det_jacobian(z, event_ndims=1)
# )


def sampler_grads_to_low_error(
    sampler, model, batch_size, key, postprocess_samples=lambda x:jnp.nanmedian(x, axis=0),
    calculate_ess_corr=False,
):

    try:
        model.sample_transformations[
            "square"
        ].ground_truth_mean, model.sample_transformations[
            "square"
        ].ground_truth_standard_deviation
    except:
        raise AttributeError("Model must have E_x2 and Var_x2 attributes")

    keys = jax.random.split(key, batch_size)

    # this key is deliberately fixed to the same value: we want the set of initial positions to be the same for different samplers
    init_keys = jax.random.split(jax.random.key(2), batch_size)

    if hasattr(model, "sample_init") and model.sample_init is not None:
        initial_position = jax.vmap(model.sample_init)(init_keys)
        jax.debug.print("using sample init {x}",x=True)
    else:
        initial_position = jax.vmap(lambda key: initialize_model(model, key))(init_keys)

    # sampler(initial_position=None,key=None)
    # from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    #     adjusted_mclmc,
    # )
    
    # samples, metadata = jax.pmap(
    #         lambda key, pos: adjusted_mclmc(num_tuning_steps=5000, return_samples=True)(
    #         model=model, num_steps=4000, initial_position=pos, key=key
    #         )
    #         )(
    #         keys,
    #         initial_position,
    #         )
    
    # error_at_each_step = get_standardized_squared_error(
    #     samples, 
    #     f=lambda x:x**2,
    #     E_f=model.sample_transformations["square"].ground_truth_mean,
    #     Var_f=model.sample_transformations["square"].ground_truth_standard_deviation**2,
    #     contract_fn=jnp.mean
    #     )
    
    # grad_evals_per_step = metadata['num_grads_per_proposal'].mean()
    # grads_to_low_avg_x2 = (
    #     samples_to_low_error(
    #         error_at_each_step,
    #     )
    #     * grad_evals_per_step
    # )
    
    # jax.debug.print("grads to low avg{x}",x=grads_to_low_avg_x2)

    # samples_to_low_err = samples_to_low_error(error_at_each_step) * gradient_calls_per_proposal

    samples, metadata = sampler(
        keys,
        initial_position,
    )

    # jax.debug.print("shape {x}",x=samples[:, 0, 0])
    jax.debug.print("shape {x}",x=samples[0, 0, :])

    if False:

        runs = jax.pmap(samples_to_low_error)(samples[:,:,1])

        failures = jnp.argwhere(jnp.isinf(runs))
        successes = jnp.argwhere(1-jnp.isinf(runs))

        num_failures = failures.shape[0]
        percent_failures = num_failures / batch_size
        good_runs = runs[successes.squeeze()]

        mean_runs = jnp.mean(good_runs)

        # jax.debug.print("\nfoo\n {x}", x=runs)
        # jax.debug.print("\nbar\n {x}", x=jnp.argwhere(jnp.isinf(runs)))
        jax.debug.print("\npercent failures\n {x}", x=percent_failures)

        # squared_errors = postprocess_samples(jnp.expand_dims(samples,0).shape)
        
        # jax.debug.print("\ngrads\n {x}", x=mean_runs * grad_evals_per_step)

        std = jnp.std(good_runs* grad_evals_per_step)

        mean_grads = mean_runs* grad_evals_per_step

        jax.debug.print("\nresult\n {x}", x=(mean_grads-std, mean_grads , mean_grads+std))

    grad_evals_per_step = metadata["num_grads_per_proposal"].mean()

    if calculate_ess_corr:

        ess_correlation_max = jnp.min(effective_sample_size(samples) / (samples.shape[1] * batch_size * grad_evals_per_step))
        ess_correlation_avg = jnp.mean(effective_sample_size(samples) / (samples.shape[1] * batch_size * grad_evals_per_step))

        # jax.debug.print("\nAVERAGE: {x}\n",x=jnp.mean(samples, axis=1))
        # jax.debug.print("\nAVERAGE: {x}\n",x=jnp.max(jnp.mean(samples, axis=1)))

        squared_errors_max_x2 = get_standardized_squared_error(samples, lambda x:x**2,
            model.sample_transformations["square"].ground_truth_mean,
            model.sample_transformations["square"].ground_truth_standard_deviation**2,
            contract_fn=jnp.max)
        squared_errors_avg_x2 = get_standardized_squared_error(samples, lambda x:x**2,
            model.sample_transformations["square"].ground_truth_mean,
            model.sample_transformations["square"].ground_truth_standard_deviation**2,
            contract_fn=jnp.mean)
        squared_errors_max_x = get_standardized_squared_error(samples, lambda x:x,
            model.sample_transformations["identity"].ground_truth_mean,
            model.sample_transformations["identity"].ground_truth_standard_deviation**2,
            contract_fn=jnp.max)
        squared_errors_avg_x = get_standardized_squared_error(samples, lambda x:x,
            model.sample_transformations["identity"].ground_truth_mean,
            model.sample_transformations["identity"].ground_truth_standard_deviation**2,
            contract_fn=jnp.mean)
        
        # jax.debug.print("squared errors {x}", x=squared_errors_max_x2[:3])
        
        squared_errors = jnp.stack([squared_errors_avg_x2, squared_errors_max_x2, squared_errors_avg_x, squared_errors_max_x], axis=1)

        # jax.debug.print("shape 1 {x}", x=squared_errors.shape)
        # jax.debug.print("shape 2 {x}", x=squared_errors_max_x2.shape)


    else:


        samples = postprocess_samples(samples)
        squared_errors = samples
        ess_correlation_max = ess_correlation_avg = jnp.nan
        # jax.debug.print("\nAVERAGE: {x}\n",x=squared_errors[-1, 2])
        # jax.debug.print("squared errors {x}", x=squared_errors[:2, 1]) 


    

    err_t_avg_x2 = (squared_errors[:, 0])
    grads_to_low_avg_x2 = (
        samples_to_low_error(
            err_t_avg_x2,
        )
        * grad_evals_per_step
    )

    err_t_max_x2 = (squared_errors[:, 1])
    grads_to_low_max_x2 = (
        samples_to_low_error(
            err_t_max_x2,
        )
        * grad_evals_per_step
    )

    err_t_avg_x = (squared_errors[:, 2])
    grads_to_low_avg_x = (
        samples_to_low_error(
            err_t_avg_x,
        )
        * grad_evals_per_step
    )

    err_t_max_x = (squared_errors[:, 3])
    grads_to_low_max_x = (
        samples_to_low_error(
            err_t_max_x,
        )
        * grad_evals_per_step
    )

    return (
        {
            "max_over_parameters": {
                "square": {
                    "error": err_t_max_x2,
                    "grads_to_low_error": grads_to_low_max_x2.item(),
                },
                "identity": {
                    "error": err_t_max_x,
                    "grads_to_low_error": grads_to_low_max_x.item(),
                },
                "autocorrelation": ess_correlation_max
            },
            "avg_over_parameters": {
                "square": {
                    "error": err_t_avg_x2,
                    "grads_to_low_error": grads_to_low_avg_x2.item(),
                },
                "identity": {
                    "error": err_t_avg_x,
                    "grads_to_low_error": grads_to_low_avg_x.item(),
                },
                "autocorrelation": ess_correlation_avg
            },
            "num_tuning_grads": metadata["num_tuning_grads"].mean().item(),
            "L": metadata["L"].mean().item(),
            "step_size": metadata["step_size"].mean().item(),
        },
        squared_errors,
    )
