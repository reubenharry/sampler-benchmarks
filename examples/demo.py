import itertools
import inference_gym.using_jax as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
sys.path.append('.')
from src.samplers import samplers
from src.models import models
from src.ess import evaluate_sampler
import pandas as pd

key = jax.random.PRNGKey(0)
results = []

for i, (sampler, model) in enumerate(itertools.product(samplers, models)):

    key = jax.random.fold_in(key, i)

    err_t_mean_max, grads_to_low_max, err_t_mean_avg, grads_to_low_avg, expectation = evaluate_sampler(sampler=samplers[sampler](),model=models[model], num_steps=100000, batch_size=128, key=key)
    
    # Append the results to the list
    results.append({
        'Sampler': sampler,
        'Model': model,
        'Grad evaluations to low error (avg)': grads_to_low_avg
    })

# Create the DataFrame
df = pd.DataFrame(results)

# Save the DataFrame
df.to_csv('results/grads_to_low_error.csv')


