import os
import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)

num_cores = jax.local_device_count()
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from src.samplers import samplers
from src.models import models
from src.ess import sampler_grads_to_low_error
import pandas as pd

key = jax.random.PRNGKey(1)
results = []

for i, (sampler, model) in enumerate(itertools.product(samplers, models)):

    key = jax.random.fold_in(key, i)

    print(f'Running sampler {sampler} on model {model}')

    (
        err_t_mean_max,
        grads_to_low_max,
        err_t_mean_avg,
        grads_to_low_avg,
        expectation,
    ) = sampler_grads_to_low_error(
        sampler=samplers[sampler](),
        model=models[model],
        num_steps=10000,
        batch_size=128,
        key=key,
        pvmap=jax.pmap
    )

    # Append the results to the list
    results.append(
        {
            "Sampler": sampler,
            "Model": model,
            "Grad evaluations to low error (avg)": grads_to_low_avg,
        }
    )

# Create the DataFrame
df = pd.DataFrame(results)

# Save the DataFrame
df.to_csv("results/grads_to_low_error.csv")
