import os
import pprint
import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)

num_cores = jax.local_device_count()
import itertools
import sys

# sys.path.append("../sampler-evaluation")
sys.path.append("..")
sys.path.append(".")
from sampler_comparison.samplers import samplers
from sampler_evaluation.models import models
from sampler_comparison.samplers.general import sampler_grads_to_low_error

from sampler_evaluation.samplers.hamiltonianmontecarlo.nuts import nuts

key = jax.random.PRNGKey(1)
results = []

sampler = samplers["nuts"]()
model = models["Banana"]


import jax.numpy as jnp

trans = model.sample_transformations["identity"](jnp.ones((32,)))

# print(trans['locs'].shape)

# raise Exception

(
    metadata,
    squared_errors,
) = sampler_grads_to_low_error(
    sampler=sampler, model=model, num_steps=5000, batch_size=2, key=key, pvmap=jax.pmap
)

print("Grads to low error for x^2 (avg across parameters)")
pprint.pprint(metadata["avg_over_parameters"]["square"]["grads_to_low_error"])

# TODOS
# - samplers:
#     mala, underdamped, ghmc with alba, nuts, unadjusted + adjusted mclmc
# - CI
# - models:
# -- finish getting long nuts run convergence metrics standardized
# -- update banana to use long nuts run
# publish package
