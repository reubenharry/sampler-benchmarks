import os
import jax
import sampler_evaluation
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()
import sys
sys.path.append(".")
sys.path.append("../sampler-evaluation")
from sampler_comparison.results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
import sampler_evaluation

run_benchmarks(
    models={
        # "Banana": banana(),
        # "Gaussian_100D": Gaussian(ndims=100),
        # "Brownian_Motion": brownian_motion(),
        "German_Credit": sampler_evaluation.models.german_credit(),
        # "Stochastic_Volatility": sampler_evaluation.models.stochastic_volatility(),
        # "Item_Response": item_response(),
    },
    samplers={
        # "unadjusted_microcanonical": unadjusted_mclmc,
        # "nuts": nuts,
        "adjusted_microcanonical": adjusted_mclmc,
    },
    batch_size=128,
    num_steps=10000,
    save_dir="sampler_comparison/results",
    key=jax.random.key(19),
)
