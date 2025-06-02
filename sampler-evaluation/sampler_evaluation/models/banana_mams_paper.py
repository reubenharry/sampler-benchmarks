from collections import namedtuple
import sys
sys.path.append(".")
sys.path.append("../sampler-evaluation")
import jax.numpy as jnp 
import jax
from sampler_evaluation.models.model import SampleTransformation, make_model

def logdensity_fn_banana(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

def exact_sample(key):
        z = jax.random.normal(key, shape=(2,))
        x0 = 10.0 * z[0]
        x1 = 0.03 * (x0**2 - 100) + z[1]
        return jnp.array([x0, x1])

# SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation"])
# Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations", "exact_sample" ])

banana_mams_paper = make_model(
        logdensity_fn=logdensity_fn_banana,
        ndims=2,
        default_event_space_bijector=lambda x:x,
        sample_transformations = {
        'square':SampleTransformation(
            fn=lambda x: x**2,
            ground_truth_mean=jnp.array([100.0, 19.0]),
            ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898])),
        )},
        sample_init= lambda key: jax.random.normal(key, shape=(2,)) * jnp.array([20.0, 10.0]),
        exact_sample=exact_sample,
        name="Banana_MAMS_Paper",
)

# banana_mams_paper = Model(
#     ndims = 2,
#     log_density_fn=logdensity_fn_banana,
#     default_event_space_bijector=transform,
#     sample_transformations={
#         "identity": SampleTransformation(ground_truth_mean=jnp.ones(2)+jnp.inf, ground_truth_standard_deviation=3.0+jnp.inf),
#         "square": SampleTransformation(ground_truth_mean=jnp.array([100.0, 19.0]), ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))),
#     },
#     exact_sample=exact_sample,
# )