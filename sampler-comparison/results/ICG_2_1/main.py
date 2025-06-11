import os 
# print(os.listdir("../../blackjax/blackjax"))
# print(os.listdir("../../blackjax"))
import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym") 
sys.path.append(".")  
# sys.path.append("../../blackjax/blackjax")
sys.path.append("../../blackjax")
sys.path.append("../../blackjax-benchmarks/sampler-comparison")
sys.path.append("../../blackjax-benchmarks/sampler-evaluation")
from results.run_benchmarks import lookup_results
import jax
batch_size = 1
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import itertools
# import pandas as pd

mh_options = [True]
canonical_options = [True]
langevin_options = [False]
tuning_options = ['nuts']
integrator_type_options = ['velocity_verlet'] # , 'mclachlan', 'omelyan']
diagonal_preconditioning_options = [True]
models = [IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log')]

redo = True 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)
