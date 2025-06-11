from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 4
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")
sys.path.append("../../blackjax")
sys.path.append("../sampler-evaluation")

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.stochastic_volatility_artificial_20000 import stochastic_volatility_artificial_20000
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc

print(stochastic_volatility_artificial_20000.sample_transformations["square"].ground_truth_mean)
print(stochastic_volatility_artificial_20000.sample_transformations["square"].ground_truth_standard_deviation)
print(stochastic_volatility_artificial_20000.sample_transformations["identity"].ground_truth_mean)
print(stochastic_volatility_artificial_20000.sample_transformations["identity"].ground_truth_standard_deviation)



run_benchmarks(
        models={
            "Stochastic_Volatility_Artificial_20000": stochastic_volatility_artificial_20000,
        },
        samplers={

            # "adjusted_microcanonical": lambda: adjusted_mclmc(num_tuning_steps=5000),
            # "adjusted_microcanonical_langevin": lambda: adjusted_mclmc(L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
            "nuts": partial(nuts,num_tuning_steps=500, progress_bar=True),
            # "unadjusted_microcanonical": lambda: unadjusted_mclmc(num_tuning_steps=25000, 
            #                                                       desired_energy_var=5e-6
            #                                                       ),
        },
        batch_size=batch_size,
        num_steps=5000,
        save_dir="results/Stochastic_Volatility_Artificial_20000",
        key=jax.random.key(19),
        map=jax.pmap
    )
