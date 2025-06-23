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


import itertools
# import pandas as pd


cos_angle_options = np.cos(np.linspace(0.1, np.pi * 0.75, 40))

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
        