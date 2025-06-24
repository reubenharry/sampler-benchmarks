<<<<<<< HEAD
import os 
# print(os.listdir("../../blackjax/blackjax"))
# print(os.listdir("../../blackjax"))
import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym") 
sys.path.append(".")  
# sys.path.append("../../blackjax/blackjax")
sys.path.append("../../blackjax")
sys.path.append("../../probability/spinoffs/inference_gym")
sys.path.append("../../blackjax-benchmarks/sampler-comparison")
sys.path.append("../../blackjax-benchmarks/sampler-evaluation")
from results.run_benchmarks import lookup_results
import jax
batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()
import numpy as np
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.rosenbrock import Rosenbrock


=======
import jax
import os 
batch_size = 512
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
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
import itertools

model = IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')

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

<<<<<<< HEAD
=======
mh_options = [True]
canonical_options = [True]
langevin_options = [False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet'] # , 'mclachlan', 'omelyan']
diagonal_preconditioning_options = [False]
models = [IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')]
# models = [banana()]
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b

cos_angle_options = np.cos(np.linspace(0.1, np.pi * 0.75, 40))

<<<<<<< HEAD
models = [
          #german_credit(), 
          #brownian_motion(), 
          #Rosenbrock(),
          IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log'),
          #, banana()
          ]

          git revert --no-commit 1ea9ac4..HEAD


for model in models:
    for cos_angle in cos_angle_options:
        print(model.name, cos_angle)
        results = lookup_results(model=model, 
                                mh=True, 
                                batch_size=batch_size,
                                canonical=True,
                                langevin=False,
                                tuning='nuts',
                                integrator_type='velocity_verlet',
                                diagonal_preconditioning=True,
                                cos_angle_termination = cos_angle)
        
=======
for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=2000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)
>>>>>>> 36f8e126e1b923da17a5f4bca28394701042723b
