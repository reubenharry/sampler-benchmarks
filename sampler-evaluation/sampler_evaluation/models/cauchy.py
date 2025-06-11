import jax
import jax.numpy as jnp

from sampler_evaluation.models.model import SampleTransformation, make_model

# class Cauchy():
#     """d indpendent copies of the standard Cauchy distribution"""

#     def __init__(self, d):
#         self.name = 'Cauchy'
#         self.ndims = d

#         self.logdensity_fn = lambda x: -jnp.sum(jnp.log(1. + jnp.square(x)))
        
#         self.transform = lambda x: x        
#         self.sample_init = lambda key: jax.random.normal(key, shape=(self.ndims,))
logdensity_fn_1D=lambda x: -jnp.log(jnp.pi*(1. + jnp.square(x)))

def cauchy(ndims):
    return make_model(
    logdensity_fn=lambda x: jnp.sum(logdensity_fn_1D(x)),
    ndims=ndims,
    default_event_space_bijector=lambda x: x,
    sample_transformations = {
        'entropy': SampleTransformation(
            fn=lambda x: -logdensity_fn_1D(x),
            # pretty_name='Entropy',
            ground_truth_mean=jnp.log(4*jnp.pi),
            ground_truth_standard_deviation=(jnp.pi**2)/3,
        ),
    },
    name=f'Cauchy_{ndims}D',

)
