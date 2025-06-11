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
import itertools


mh_options = [True]
canonical_options = [True]
langevin_options = [False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet'] # , 'mclachlan', 'omelyan']
diagonal_preconditioning_options = [False]
models = [IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')]
# models = [banana()]

redo = False 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=1000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)
