import os
import jax

import sampler_evaluation

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)

num_cores = jax.local_device_count()
import sys

# sys.path.append("..")
sys.path.append(".")


from sampler_comparison.results.run_benchmarks import run_benchmarks
from sampler_evaluation.models.banana import banana


run_benchmarks(models={"Banana": banana()}, batch_size=128, num_steps=10000, save_dir="sampler_comparison/results")
