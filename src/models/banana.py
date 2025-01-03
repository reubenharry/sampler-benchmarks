
# a class that inherits from the inference_gym Model class but adds expectations of x^2

import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp

def banana(): 
    banana = gym.targets.Banana()
    banana.sample_transformations["square"] = model.Model.SampleTransformation(
                  fn=lambda params: params**2,
                  pretty_name='Square',
                  ground_truth_mean=jnp.array([100.0, 19.0]),
                  ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))
              )
    
    banana.ndims = 2
    
    return banana

# print(banana.sample_transformations)

# print(banana.default_event_space_bijector(jnp.array([100.0, 19.0])))