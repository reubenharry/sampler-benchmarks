
import os
import jax
import jax.numpy as jnp
from sampler_evaluation.models.model import SampleTransformation, make_model
from jax.scipy.special import gamma


def StudentT(ndims, dof):

    e_x2 = jnp.ones(ndims) * dof / (dof - 2.)
    
    sample_init = lambda key: jax.random.t(key, df= dof, shape= (ndims,))

    sample_transformations = {  
        "identity": SampleTransformation(
            fn=lambda params: params,
            ground_truth_mean=jnp.zeros(ndims),
            ground_truth_standard_deviation=jnp.sqrt(e_x2)), 
        # "covariance": SampleTransformation(
        #     fn=lambda params: jnp.outer(params, params),
        #     name="Covariance",
        #     ground_truth_mean= jnp.diagonal(e_x2),
        #     ground_truth_standard_deviation=jnp.nan)
    }
    
    model = make_model(
            logdensity_fn= lambda x: - 0.5 * (dof + 1) * jnp.sum(jnp.log(1. + jnp.square(x) / dof)),
            ndims=ndims,
            default_event_space_bijector= lambda x: x,
            sample_transformations = sample_transformations,
            exact_sample= sample_init,
            name=f"Student-t_dof={dof}",
            sample_init = sample_init
            )
    
    inv_cov = jnp.diag(1./e_x2)

    return model, inv_cov



def Power(ndims, p):

    e_x2 = jnp.ones(ndims) * gamma(3./p) / gamma(1./p)
    
    sample_init = lambda key: jax.random.normal(key, shape=(ndims,))

    sample_transformations = {  
        "identity": SampleTransformation(
            fn=lambda params: params,
            ground_truth_mean=jnp.zeros(ndims),
            ground_truth_standard_deviation=jnp.sqrt(e_x2))
    }
    
    model = make_model(
            logdensity_fn= lambda x: - jnp.sum(jnp.power(jnp.abs(x), p)),
            ndims=ndims,
            default_event_space_bijector= lambda x: x,
            sample_transformations = sample_transformations,
            name=f"Power_{p}",
            sample_init = sample_init
            )
    
    inv_cov = jnp.diag(1./e_x2)

    return model, inv_cov