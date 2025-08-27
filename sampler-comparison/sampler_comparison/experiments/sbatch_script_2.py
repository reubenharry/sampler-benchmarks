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
from sampler_comparison.experiments.benchmark import clear_jax_cache, run
from sampler_evaluation.models.cauchy import cauchy


if __name__ == "__main__":

    


    dims = np.concatenate([np.ceil(np.logspace(6,7, 4)).astype(int)])[:]
    models = [IllConditionedGaussian(ndims=dim, condition_number=1, eigenvalues='log', do_covariance=False) for dim in dims]


    # print(dims)
    # raise Exception("Stop here")


    # dims = 2*np.concatenate([np.ceil(np.logspace(2,5, 10)).astype(int)])[:]
    # models = [Rosenbrock(D=dim) for dim in dims]
    
    for model in models:
        

            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['nuts'],
            #     mh_options = [True],
            #     canonical_options = [True],
            #     langevin_options = [False],
            #     integrator_type_options = ['velocity_verlet', 'mclachlan'],
            #     diagonal_preconditioning_options = [False],
            #     redo=True,
            #     # redo_bad_results=True
            # )

            run(
                key=jax.random.PRNGKey(4),
                models=[model],
                tuning_options=['alba'],
                mh_options = [True, False],
                canonical_options = [True, False],
                langevin_options = [True, False],
                integrator_type_options = ['velocity_verlet', 'mclachlan', 'omelyan'],
                diagonal_preconditioning_options = [False],
                redo=False,
                compute_missing=True,
                # redo_bad_results=True
            )
            
            # Clear cache after each model to prevent memory accumulation
            print(f"\n=== Completed model: {model.name} ===")
            print("Clearing JAX cache after model completion...")
            clear_jax_cache()
            
            # run(
            #     key=jax.random.PRNGKey(5),
            #     models=[model],
            #     tuning_options=['alba'],
            #     mh_options = [True, False],
            #     canonical_options = [True, False],
            #     langevin_options = [True, False],
            #     integrator_type_options = ['velocity_verlet', 'mclachlan'],
            #     diagonal_preconditioning_options = [True, False],
            #     redo=False,
            #     compute_missing=True,
            # )
            
            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['nuts'],
            #     mh_options = [True],
            #     canonical_options = [True],
            #     langevin_options = [False],
            #     integrator_type_options = ['velocity_verlet', 'mclachlan'],
            #     diagonal_preconditioning_options = [True, False],
            #     redo=True,
            #     # redo_bad_results=True
            # )

        # run(
        #     models=models,
        #     tuning_options=['grid_search'],
        #     mh_options = [False],
        #     canonical_options = [True],
        #     langevin_options = [True, False],
        #     integrator_type_options = ['mclachlan'],
        #     diagonal_preconditioning_options = [True],
        #     redo=False,
        #     # mh_options = [False],
        #     # diagonal_preconditioning_options = [False],
        #     # redo_bad_results=True
        # )

      