import jax.interpreters.xla as xla
import jax.core

# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # defrags GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"      # don't grab all VRAM up front
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
import os
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../blackjax')
sys.path.append('../sampler-comparison')
sys.path.append('../sampler-evaluation')
sys.path.append('../')
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sampler_evaluation
from sampler_comparison.samplers import samplers
import seaborn as sns
from functools import partial
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import initialize_model
from sampler_evaluation.models.banana import banana
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc


from sampler_comparison.samplers.general import make_log_density_fn
import blackjax
from sampler_evaluation.evaluation.ess import samples_to_low_error, get_standardized_squared_error

import numpy as np
import jax
import jax.numpy as jnp
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc_no_tuning
from results.run_benchmarks import lookup_results
import jax.interpreters.xla as xla
import jax.core
if not hasattr(xla, "pytype_aval_mappings"):
    xla.pytype_aval_mappings = jax.core.pytype_aval_mappings



# print(os.listdir("../../../sampler-comparison"))

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable preallocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
import numpy as np

import pandas as pd
import os
import jax
from results.run_benchmarks import lookup_results
import pandas as pd
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_comparison.experiments.benchmark import run

if __name__ == "__main__":

    

        
    models = [
        IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        ]

   
    for model in models:
       run(
                key=jax.random.PRNGKey(4),
                models=[model],
                tuning_options=['alba'],
                mh_options = [True],
                canonical_options = [True],
                langevin_options = [False],
                integrator_type_options = ['velocity_verlet'],
                diagonal_preconditioning_options = [True],
                redo=True,
                compute_missing=True,
                redo_bad_results=True,
                pseudofermion=False,
            )
            
         