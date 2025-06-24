# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

# print(os.listdir("../../../sampler-comparison"))

# import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
# raise Exception("stop")

from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from functools import partial
# import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import pandas as pd
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
import os
from results.run_benchmarks import run_benchmarks
# import sampler_evaluation
from sampler_evaluation.models.dirichlet import Dirichlet
import jax
from results.run_benchmarks import lookup_results
import itertools
import jax.numpy as jnp
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.item_response import item_response
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_comparison.experiments.utils import model_info
from sampler_comparison.experiments.plotting import plot_results
from sampler_evaluation.models.u1 import U1
import time 

def run(models, mh_options=[True, False], canonical_options=[True, False], langevin_options=[True, False], tuning_options=['alba'], integrator_type_options=['velocity_verlet','mclachlan'], diagonal_preconditioning_options=[True, False], redo=False, compute_missing=True, redo_bad_results=None):

    
    # mh_options = [True,False]
    # canonical_options = [True,False]
    # langevin_options = [True,False]
    # tuning_options = ['alba']
    # integrator_type_options = ['velocity_verlet','mclachlan']
    # diagonal_preconditioning_options = [True,False]
    
   


    


    full_results = pd.DataFrame()
    for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
        time_start = time.time()
        results = lookup_results(
            model=model, 
            num_steps=model_info[model.name]['num_steps'][mh], 
            mh=mh, 
            canonical=canonical, 
            langevin=langevin, 
            tuning=tuning, 
            integrator_type=integrator_type, 
            diagonal_preconditioning=diagonal_preconditioning, 
            redo=redo, 
            batch_size=model_info[model.name]['batch_size'], 
            relative_path='./', 
            compute_missing=compute_missing,
            redo_bad_results=redo_bad_results
        )
        full_results = pd.concat([full_results, results], ignore_index=True)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
   
if __name__ == "__main__":

    # run(
    #     models=[IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log', do_covariance=False)], 
    #     redo=True)

    # run(
    #     models=[U1(Lt=16, Lx=16, beta=6)], 
    #     mh_options = [True],
    #     canonical_options = [True],
    #     langevin_options = [False],
    #     tuning_options = ['nuts'],
    #     integrator_type_options = ['velocity_verlet'],
    #     diagonal_preconditioning_options = [False], redo=True)

    from sampler_evaluation.models.cauchy import cauchy


    # run(
    #     models=[cauchy(ndims=100)],
    #     mh_options = [False],
    #     diagonal_preconditioning_options = [False],
    #     redo=False,
    #     redo_bad_results=True
    # )

    models = [
        # IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        # IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log', do_covariance=False),
        IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log', do_covariance=False),
        # item_response(),
        # IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log', do_covariance=False),
        # brownian_motion(),
        # german_credit(),
        # banana(),
        # Rosenbrock(18),
        # stochastic_volatility_mams_paper,
        # U1(Lt=16, Lx=16, beta=6)
        ]
    
    # run(
    #     models=models,
    #     tuning_options=['alba'],
    #     mh_options = [True, False],
    #     canonical_options = [False],
    #     langevin_options = [True, False],
    #     integrator_type_options = ['velocity_verlet', 'mclachlan'],
    #     diagonal_preconditioning_options = [True, False],
    #     redo=True,
    #     # mh_options = [False],
    #     # diagonal_preconditioning_options = [False],
    #     # redo_bad_results=True
    # )

    # print(IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log').sample_transformations)
    # raise Exception("stop")
    
    # run(
    #     models=[IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log', do_covariance=False)],
    #     tuning_options=['alba'],
    #     mh_options = [True],
    #     canonical_options = [False],
    #     langevin_options = [True, False],
    #     integrator_type_options = ['velocity_verlet', 'mclachlan'],
    #     diagonal_preconditioning_options = [True, False],
    #     redo=False,
    #     # mh_options = [False],
    #     # diagonal_preconditioning_options = [False],
    #     redo_bad_results='avg'
    # )

    run(
        models=[IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log', do_covariance=False)],
        tuning_options=['grid_search'],
        mh_options = [False],
        canonical_options = [True, False],
        langevin_options = [True],
        integrator_type_options = ['velocity_verlet', 'mclachlan'],
        diagonal_preconditioning_options = [True, False],
        # redo=True,
        # mh_options = [False],
        # diagonal_preconditioning_options = [False],
        # redo_bad_results=True
    )

    # plot_results()
    # plot_results()