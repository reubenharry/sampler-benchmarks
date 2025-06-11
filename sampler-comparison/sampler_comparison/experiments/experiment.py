from collections import namedtuple
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys

current_path = os.getcwd()
sys.path.append(current_path + '/../../blackjax/')
sys.path.append(current_path + '/../sampler-evaluation/')

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.rosenbrock import Rosenbrock
# from sampler_evaluation.models.standardgaussian import Gaussian
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.neals_funnel import neals_funnel
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.neals_funnel_mams_paper import neals_funnel_mams_paper
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper




# 64 bit
with jax.experimental.enable_x64():
    run_benchmarks(
        models={
            #"Banana": banana(),
            #"Banana_MAMS": banana_mams_paper,
            "Gaussian_100D": IllConditionedGaussian(ndims=100, condition_number=10000, eigenvalues='log'),
            # "Rosenbrock": Rosenbrock(),
            # "Neals_Funnel": sampler_evaluation.models.neals_funnel(),
            #"Neals_Funnel_MAMS": neals_funnel_mams_paper,
            # "Brownian_Motion": brownian_motion(),
            # "German_Credit": sampler_evaluation.models.german_credit(),
            # "Stochastic_Volatility": sampler_evaluation.models.stochastic_volatility(),
            # "Stochastic_Volatility_MAMS": stochastic_volatility_mams_paper,
            # "Item_Response": item_response(),
        },
        samplers={
            # "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=10000),
            #"adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=20000, do_nuts_warmup=False, target_acc_rate=0.99),
            #"adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=20000, do_nuts_warmup=False, target_acc_rate=0.99),
            # "unadjusted_microcanonical": unadjusted_mclmc,
            "nuts": lambda: nuts(num_tuning_steps=5000),
        },
        batch_size=batch_size,
        num_steps=100000,
        save_dir="sampler_comparison/results",
        key=jax.random.key(19),
        map=jax.pmap
    )

# condition number 1: adjusted: 3573, nuts: 19000
# condition number 10000: adjusted (nuts 500): 36592