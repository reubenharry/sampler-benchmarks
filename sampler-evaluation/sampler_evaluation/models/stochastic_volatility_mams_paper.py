import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from collections import namedtuple
import sampler_evaluation

# batch_size = 32
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
# num_cores = jax.local_device_count()



def nlogp_StudentT(x, df, scale):
    y = x / scale
    z = (
        jnp.log(scale)
        + 0.5 * jnp.log(df)
        + 0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln(0.5 * df)
        - jax.scipy.special.gammaln(0.5 * (df + 1.0))
    )
    return 0.5 * (df + 1.0) * jnp.log1p(y**2.0 / df) + z

name = 'StochasticVolatility'

typical_sigma, typical_nu = 0.02, 10.0

ndims = 2429

E_x2, Var_x2 = jnp.load('../sampler-evaluation/sampler_evaluation/models/data/stoch_vol_moments.npy')
SP500_returns = jnp.load('../sampler-evaluation/sampler_evaluation/models/data/' + 'SP500.npy')  

def logdensity_fn(x):
        """x=  [s1, s2, ... s2427, log sigma / typical_sigma, log nu / typical_nu]"""

        sigma = jnp.exp(x[-2]) * typical_sigma #we used this transformation to make x unconstrained
        nu = jnp.exp(x[-1]) * typical_nu

        l1= (jnp.exp(x[-2]) - x[-2]) + (jnp.exp(x[-1]) - x[-1])
        l2 = (ndims - 2) * jnp.log(sigma) + 0.5 * (jnp.square(x[0]) + jnp.sum(jnp.square(x[1:-2] - x[:-3]))) / jnp.square(sigma)
        l3 = jnp.sum(nlogp_StudentT(SP500_returns, nu, jnp.exp(x[:-2])))

        return -(l1 + l2 + l3)

def transform(x):
        """transforms to the variables which are used by numpyro"""

        z = jnp.empty(x.shape)
        z = z.at[:-2].set(x[:-2]) # = s = log R
        z = z.at[-2].set(jnp.exp(x[-2]) * typical_sigma) # = sigma
        z = z.at[-1].set(jnp.exp(x[-1]) * typical_nu) # = nu

        return z

SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation"])
Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations", "name" ])

stochastic_volatility_mams_paper = Model(
    ndims = ndims,
    log_density_fn=logdensity_fn,
    default_event_space_bijector=transform,
    sample_transformations={
        "identity": SampleTransformation(ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf),
        "square": SampleTransformation(ground_truth_mean=E_x2, ground_truth_standard_deviation=jnp.sqrt(Var_x2)),
    },
    name="StochasticVolatility_MAMS_Paper"
)