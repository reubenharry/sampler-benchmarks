"""
This script is used to estimate the expectations of the model using a long NUTS run.
We recommend running this script on GPU: by default the script runs 4 chains, in parallel, on 4 GPUs.
"""

# TODO: # The script also returns a diagnostic, namely the potential scale reduction factor, which should be close to

import itertools
import pickle
import sys
import numpy as np


sys.path.append("./")
sys.path.append("../blackjax")
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(4)


from src.models import models
from src.samplers import samplers
import jax
import jax.numpy as jnp
import blackjax
import time
from blackjax.diagnostics import potential_scale_reduction


num_chains = 4

gold_standard_expectation_steps = {"Banana": 1000000}


def relative_fluctuations(expectation):
    expectation = expectation.T
    expectation_mean = jnp.mean(expectation, axis=1)
    diff = jnp.abs((expectation - expectation_mean[:, None]) / expectation_mean[:, None])
    return jnp.max(diff)



def nuts_rhat(model):

    sampler = samplers["nuts"](
        integrator_type="velocity_verlet",
        diagonal_preconditioning=True,
        return_samples=False,
        incremental_value_transform=lambda x: x,
        num_tuning_steps=5000,
    )

    key = jax.random.PRNGKey(1)
    init_key, run_key = jax.random.split(key)
    run_keys = jax.random.split(run_key, num_chains)
    init_keys = jax.random.split(init_key, num_chains)
    init_keys = jax.random.split(init_key, num_chains)
    init_pos = jax.pmap(lambda key: models[model].sample(seed=key))(init_keys)

    expectation, metadata = jax.pmap(
        lambda pos, key: sampler(
            model=models[model],
            num_steps=gold_standard_expectation_steps[model],
            initial_position=pos,
            key=key,
        )
    )(init_pos, run_keys)

    expectation = np.array(expectation)

    e_x = expectation[:, -1, 0, :]
    e_x2 = expectation[:, -1, 1, :]
    e_x4 = expectation[:, -1, 2, :]

    e_x_avg = e_x.mean(axis=0)
    e_x2_avg = e_x2.mean(axis=0)
    e_x4_avg = e_x4.mean(axis=0)

    # print("potential scale reduction", (potential_scale_reduction(e_x2)))
    # print("relative fluctuations", relative_fluctuations(e_x2))
    # print(f"x^2 is {e_x2_avg} and var_x = {e_x2_avg - e_x_avg**2}")
    # print(f"var x^2 is {e_x4_avg - e_x2_avg**2}")

    jax.debug.print("e_x {x}", x=e_x)

    results = {
        "e_x": e_x,
        "e_x2": e_x2,
        "e_x4": e_x4,
        "potential_scale_reduction": (
            potential_scale_reduction(e_x),
            potential_scale_reduction(e_x2),
            potential_scale_reduction(e_x4),
        ),
        "relative_fluctuations": (
            relative_fluctuations(e_x),
            relative_fluctuations(e_x2),
            relative_fluctuations(e_x4),
        ),
    }

    ## pickle to data
    with open(f"./src/models/data/{models[model].name}_expectations.pkl", "wb") as f:
        pickle.dump(results, f)


toc = time.time()
(nuts_rhat("Banana"))
tic = time.time()
print(f"time: {tic-toc}")
