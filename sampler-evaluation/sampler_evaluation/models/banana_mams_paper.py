from collections import namedtuple
import sys
sys.path.append(".")
sys.path.append("../sampler-evaluation")
import jax.numpy as jnp 
import jax

def logdensity_fn_banana(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

def transform(x):
        return x

def exact_sample(key):
        z = jax.random.normal(key, shape=(2,))
        x0 = 10.0 * z[0]
        x1 = 0.03 * (x0**2 - 100) + z[1]
        return jnp.array([x0, x1])

SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation"])
Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations", "exact_sample" ])
banana_mams_paper = Model(
    ndims = 2,
    log_density_fn=logdensity_fn_banana,
    default_event_space_bijector=transform,
    sample_transformations={
        "identity": SampleTransformation(ground_truth_mean=jnp.ones(2)+jnp.inf, ground_truth_standard_deviation=3.0+jnp.inf),
        "square": SampleTransformation(ground_truth_mean=jnp.array([100.0, 19.0]), ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))),
    },
    exact_sample=exact_sample,
)