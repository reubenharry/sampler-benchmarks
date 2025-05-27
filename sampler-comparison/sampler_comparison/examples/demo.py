# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts

from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from functools import partial
# import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import pandas as pd
from sampler_evaluation.models.banana import banana
import os

model = IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log')



def lookup_results(model, mh : bool, canonical : bool, langevin : bool, tuning : str):

    # make a dictionary that maps choice of parameters to a sampler. e.g. mh=True, canonical=True, tuning='none' -> 'nuts'
    sampler_dict = {(True, True, False, 'nuts'): 'nuts',
                    (True, True, 'tuning'): 'adjusted_hmc',
                    (True, False, False, 'alba') : ('adjusted_microcanonical', partial(adjusted_mclmc,num_tuning_steps=5000))
                    # (True, False, 'none'): 'adjusted_microcanonical_langevin',
                    # (True, False, 'tuning'): 'adjusted_microcanonical_langevin',
                    # (False, True, 'none'): 'adjusted_hmc',
                    # (False, True, 'tuning'): 'adjusted_hmc',
                    }
    
    # load results
    results_dir = f'results/{model.name}'
    results = pd.read_csv(os.path.join(results_dir, f'{sampler_dict.get((mh, canonical, langevin, tuning), "None")[0]}_{model.name}.csv'))
   
    print(results)

    # sampler_dict = {(True, True)}

lookup_results(model, True, False, False, 'alba')