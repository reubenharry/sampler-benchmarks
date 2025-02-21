import os
import jax
import sampler_evaluation

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")
sys.path.append("../sampler-evaluation")
from sampler_comparison.results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.rosenbrock import Rosenbrock_36D
from sampler_evaluation.models.standardgaussian import Gaussian
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp 
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.neals_funnel import neals_funnel
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.ill_conditioned_gaussian import IllConditionedGaussian

with jax.experimental.enable_x64():

    run_benchmarks(
        models={
            "Banana": banana(),
            # "Gaussian_100D": IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
            # "Rosenbrock_36D": Rosenbrock_36D(),
            # "Neals_Funnel": sampler_evaluation.models.neals_funnel(),
            # "Brownian_Motion": brownian_motion(),
            # "German_Credit": sampler_evaluation.models.german_credit(),
            # "Stochastic_Volatility": sampler_evaluation.models.stochastic_volatility(),
            # "Item_Response": item_response(),
        },
        samplers={
            # "adjusted_microcanonical": lambda: adjusted_mclmc(), # adjusted_mclmc(num_tuning_steps=1000),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23),
            # "unadjusted_microcanonical": unadjusted_mclmc,
            "nuts": nuts,
        },
        batch_size=batch_size,
        num_steps=2000,
        save_dir="sampler_comparison/results",
        key=jax.random.key(22),
    )
