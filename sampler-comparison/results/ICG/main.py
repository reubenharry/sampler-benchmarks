import jax
import os 
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()
import sys
sys.path.append(".")  
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))

from results.run_benchmarks import lookup_results
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
# from sampler_evaluation.models.banana import banana
import itertools

# model = IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')

import jax.numpy as jnp
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc

# initial_position = jnp.array([0.0, 0.0])

# with jax.disable_jit():
#     samples, _ = unadjusted_mclmc(
#                 # initial_state=initial_state,
#                 integrator_type='velocity_verlet',
#                 num_tuning_steps=2000,
#                 diagonal_preconditioning=False,
#                 # step_size=step_size,
#                 # L=nleaps*step_size,
#                 # step_size=step_size,
#                 # L=step_size,
#                 return_samples=False,
#                 # inverse_mass_matrix=inverse_mass_matrix,
#                 )(
#                 model=model, 
#                 num_steps=4096,
#                 initial_position=initial_position, 
#                 key=jax.random.key(0),
#                 )

# raise Exception("stop")

mh_options = [True, False]
canonical_options = [True, False]
langevin_options = [True, False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet', 'mclachlan'] # , 'omelyan']
diagonal_preconditioning_options = [True, False]
models = [IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log', do_covariance=False)]
# models = [banana()]

redo = False 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=10000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)
