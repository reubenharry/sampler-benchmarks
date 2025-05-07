import os
import pickle
import jax
#from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
#import sampler_evaluation
from collections import namedtuple
import jax.numpy as jnp
from sampler_evaluation.models.model import SampleTransformation, make_model

import os
module_dir = os.path.dirname(os.path.abspath(__file__))


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

name = 'Stochastic_Volatility_MAMS_Paper'

typical_sigma, typical_nu = 0.02, 10.0

ndims = 2429

E_x2, Var_x2 = jnp.load(f'{module_dir}/data/stoch_vol_moments.npy')
SP500_returns = jnp.load(f'{module_dir}/data/' + 'SP500.npy')  

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

# SampleTransformation = namedtuple("SampleTransformation", ["ground_truth_mean", "ground_truth_standard_deviation"])
# Model = namedtuple("Model", ["ndims", "log_density_fn", "default_event_space_bijector", "sample_transformations", "name" ])

# stochastic_volatility_mams_paper = Model(
#     ndims = ndims,
#     log_density_fn=logdensity_fn,
#     default_event_space_bijector=transform,
#     sample_transformations={
#         "identity": SampleTransformation(ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf),
#         "square": SampleTransformation(ground_truth_mean=E_x2, ground_truth_standard_deviation=jnp.sqrt(Var_x2)),
#     },
#     name="StochasticVolatility_MAMS_Paper"
# )

# with open(
#         f"{module_dir}/data/{name}_expectations.pkl",
#         "rb",
#     ) as f:
#         stats = pickle.load(f)

# e_x = stats["identity"]
# cov = stats["covariance"]

stochastic_volatility_mams_paper = make_model(
        logdensity_fn=logdensity_fn,
        ndims=ndims,
        default_event_space_bijector=transform,
        sample_transformations = {
               
        "identity": SampleTransformation(
               fn=lambda x: x,
               ground_truth_mean=E_x2+jnp.inf, ground_truth_standard_deviation=jnp.sqrt(Var_x2)+jnp.inf),

        "square": SampleTransformation(
               fn=lambda x: x**2,
               ground_truth_mean=E_x2, ground_truth_standard_deviation=jnp.sqrt(Var_x2)),

 

        "quartic" : SampleTransformation(
                fn=lambda params: (params)** 4,
                ground_truth_mean=jnp.nan,
                ground_truth_standard_deviation=jnp.nan,
        ),



        # "covariance" : 
        #         SampleTransformation(
        #         fn=lambda params: jnp.outer((params) - e_x, (params) - e_x),
        #         ground_truth_mean=cov,
        #         ground_truth_standard_deviation=jnp.nan,
        #         )
        },

        exact_sample=None,
        name="Stochastic_Volatility_MAMS_Paper",
)