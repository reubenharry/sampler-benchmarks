from collections import namedtuple
import os
import jax
jax.config.update("jax_enable_x64", True)
import sampler_evaluation

batch_size = 256
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys

sys.path.append(".")
sys.path.append("../sampler-evaluation")
from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.rosenbrock import Rosenbrock_36D
from sampler_evaluation.models.standardgaussian import Gaussian
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.neals_funnel import neals_funnel
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.ill_conditioned_gaussian import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.neals_funnel_mams_paper import neals_funnel_mams_paper
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_comparison.samplers.parallel.microcanonicalmontecarlo.emaus import parallel_microcanonical

from sampler_comparison.samplers.general import sampler_grads_to_low_error

from sampler_evaluation.evaluation.ess import get_standardized_squared_error

mesh = jax.sharding.Mesh(jax.devices(), 'chains')


parallel_microcanonical(num_steps1=100, num_steps2=400, num_chains=batch_size,mesh=mesh)(
                model=brownian_motion(), num_steps=None, initial_position=None, key=jax.random.key(0)
                )





# samp = lambda key, pos: parallel_microcanonical(num_steps1=100, num_steps2=400, num_chains=batch_size,mesh=mesh)(
#                 model=banana_mams_paper, num_steps=None, initial_position=pos, key=key
#                 )

# sampler_grads_to_low_error(
#     sampler=samp,
#     model=banana_mams_paper,
#     num_steps=0,
#     batch_size=batch_size,
#     key=jax.random.key(19),
#     postprocess_samples=lambda samples: 
    
#         jnp.array(
#             [
#                 get_standardized_squared_error(
#                     jnp.expand_dims(samples,0), 
#                     lambda x:x**2,
#                     banana_mams_paper.sample_transformations["square"].ground_truth_mean,
#                     banana_mams_paper.sample_transformations["square"].ground_truth_standard_deviation**2,
#                     contract_fn=jnp.average
#                     ),

#                 get_standardized_squared_error(
#                     jnp.expand_dims(samples,0), 
#                     lambda x:x**2,
#                     banana_mams_paper.sample_transformations["square"].ground_truth_mean,
#                     banana_mams_paper.sample_transformations["square"].ground_truth_standard_deviation**2,
#                     ),
#                 get_standardized_squared_error(
#                     jnp.expand_dims(samples,0), 
#                     lambda x:x,
#                     banana_mams_paper.sample_transformations["identity"].ground_truth_mean,
#                     banana_mams_paper.sample_transformations["identity"].ground_truth_standard_deviation**2,
#                     contract_fn=jnp.average
#                     ),
#                 get_standardized_squared_error(
#                     jnp.expand_dims(samples,0), 
#                     lambda x:x,
#                     banana_mams_paper.sample_transformations["identity"].ground_truth_mean,
#                     banana_mams_paper.sample_transformations["identity"].ground_truth_standard_deviation**2,
#                     ),
                
#         ]).T
# )

# 64 bit
# with jax.experimental.enable_x64():
#     run_benchmarks(
#         models={
#             # "Banana": banana(),
#             "Banana_MAMS": banana_mams_paper,
#             # "Gaussian_100D": IllConditionedGaussian(ndims=100, condition_number=10000, eigenvalues='log'),
#             # "Rosenbrock_36D": Rosenbrock_36D(),
#             # "Neals_Funnel": sampler_evaluation.models.neals_funnel(),
#             # "Neals_Funnel_MAMS": neals_funnel_mams_paper,
#             # "Brownian_Motion": brownian_motion(),
#             # "German_Credit": sampler_evaluation.models.german_credit(),
#             # "Stochastic_Volatility": sampler_evaluation.models.stochastic_volatility(),
#             # "Stochastic_Volatility_MAMS": stochastic_volatility_mams_paper,
#             # "Item_Response": item_response(),
#         },
#         samplers={
#             # "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=10000),
#             "emaus": parallel_microcanonical
#         },
#         batch_size=batch_size,
#         num_steps=1000,
#         save_dir="sampler_comparison/results",
#         key=jax.random.key(19),
#         map=jax.pmap
#     )
