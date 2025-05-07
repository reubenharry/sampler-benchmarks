from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
import sampler_evaluation 
import itertools
import os
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
import jax
num_cores = jax.local_device_count()


import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
sys.path.append('./sampler-comparison')
sys.path.append('../../')
from sampler_comparison.samplers import samplers
import seaborn as sns

from sampler_comparison.samplers.general import initialize_model
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

from blackjax.diagnostics import effective_sample_size
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian



from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper



def ess(model, sampler, contract_fn, batch_size=128):

    # batch_size = 128
    init_keys = jax.random.split(jax.random.key(3), batch_size)
    keys = jax.random.split(jax.random.key(3), batch_size)
    initial_position = jax.vmap(lambda key: initialize_model(model, key))(init_keys)
    num_steps = 20000

    samples, metadata = jax.pmap(
            lambda key, pos: samplers[sampler](return_samples=True)(
            model=model, num_steps=num_steps, initial_position=pos, key=key
            )
            )(
            keys,
            initial_position,
            )
    
    error_at_each_step = get_standardized_squared_error(
        samples, 
        f=lambda x:x**2,
        E_f=model.sample_transformations["square"].ground_truth_mean,
        Var_f=model.sample_transformations["square"].ground_truth_standard_deviation**2,
        contract_fn=contract_fn[1]
        )

    gradient_calls_per_proposal = metadata['num_grads_per_proposal'].mean()
    samples_to_low_err = samples_to_low_error(error_at_each_step) * gradient_calls_per_proposal
    ess_correlation = contract_fn[0](effective_sample_size(samples) / (num_steps * batch_size * gradient_calls_per_proposal))
    ess_bias = 100 / samples_to_low_err
    return ess_correlation, ess_bias, samples_to_low_err

models = [
    sampler_evaluation.models.item_response(),
    stochastic_volatility_mams_paper,
    ]

for model, sampler in itertools.product(models, ['nuts', 'adjusted_microcanonical']):
    ess_correlation, ess_bias, samples_to_low_err = ess(model, sampler, contract_fn=(jnp.mean,jnp.mean), batch_size=128)
    print(f"ESS CORR of {sampler} on {model.name} is {ess_correlation}")

    print(f"ESS BIAS of {sampler} on {model.name} is {ess_bias}")