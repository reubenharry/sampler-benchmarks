import jax
import jax.numpy as jnp

from sampler_evaluation.models.model import make_model
# class Cauchy():
#     """d indpendent copies of the standard Cauchy distribution"""

#     def __init__(self, d):
#         self.name = 'Cauchy'
#         self.ndims = d

#         self.logdensity_fn = lambda x: -jnp.sum(jnp.log(1. + jnp.square(x)))
        
#         self.transform = lambda x: x        
#         self.sample_init = lambda key: jax.random.normal(key, shape=(self.ndims,))

def cauchy(ndims):
    return make_model(
    logdensity_fn=lambda x: -jnp.sum(jnp.log(1. + jnp.square(x))),
    ndims=ndims,
    default_event_space_bijector=lambda x: x,
    sample_transformations = {},
    name='Cauchy',

)
