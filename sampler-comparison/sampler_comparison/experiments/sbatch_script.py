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
from sampler_comparison.experiments.benchmark import run
from sampler_evaluation.models.cauchy import cauchy


if __name__ == "__main__":

    models = [
        brownian_motion(),
        Rosenbrock(18),
        german_credit(),
        # # IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log', do_covariance=False),
        # IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log', do_covariance=False),
        # IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log', do_covariance=False),
        # item_response(),
        # stochastic_volatility_mams_paper,
        # # U1(Lt=16, Lx=16, beta=6)
        # cauchy(ndims=100),
        # banana(),
        ]
  

    run(
        models=models,
        tuning_options=['grid_search'],
        mh_options = [True, False],
        canonical_options = [True, False],
        langevin_options = [True, False],
        integrator_type_options = ['velocity_verlet', 'mclachlan'],
        diagonal_preconditioning_options = [True, False],
        redo=False,
        key=jax.random.key(0),
        # mh_options = [False],
        # diagonal_preconditioning_options = [False],
        # redo_bad_results=True
    )
