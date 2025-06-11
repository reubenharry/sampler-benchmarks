import jax
import jax.numpy as jnp

from sampler_evaluation.models.model import SampleTransformation, make_model



def bimodal_gaussian(ndims=50):
        
    mu = 4
    sigma = 0.6
    a= 0.25

    ndims = ndims

    mu1 = jnp.zeros(ndims)
    mu2 = jnp.insert(jnp.zeros(ndims - 1), 0, mu)
    sigma1, sigma2 = 1., sigma


    # ground truth moments
    Ex2_1 = (1 - a) + a * (mu**2 + sigma**2)
    Ex2_others = (1 - a) + a * sigma**2

    Varx2_1 = (1 - a) * 3 + a * (mu**4 + 6 * mu**2 * sigma**2 + 3 * sigma**4) - Ex2_1**2
    Varx2_others = (1 - a) * 3 + a * 3 * sigma**4 - Ex2_others**2

    e_x2 = jnp.insert(jnp.ones(ndims-1) * Ex2_others, 0, Ex2_1)
    var_x2 = jnp.insert(jnp.ones(ndims-1) * Varx2_others, 0, Varx2_1)



    def logdensity_fn(x):
        """- log p of the target distribution"""

        N1 = (1.0 - a) * jnp.exp(-0.5 * jnp.sum(jnp.square(x - mu1), axis= -1) / sigma1 ** 2) / jnp.power(2 * jnp.pi * sigma1 ** 2, ndims * 0.5)
        N2 = a * jnp.exp(-0.5 * jnp.sum(jnp.square(x - mu2), axis= -1) / sigma2 ** 2) / jnp.power(2 * jnp.pi * sigma2 ** 2, ndims * 0.5)

        return jnp.log(N1 + N2)


    

      

    return make_model(
    logdensity_fn=logdensity_fn,
    ndims=ndims,
    default_event_space_bijector=lambda x: x,
    sample_transformations = {
        'square': SampleTransformation(
            fn=lambda x: x**2,
            ground_truth_mean=e_x2,
            ground_truth_standard_deviation=jnp.sqrt(var_x2),
        ),
    },
    name='BimodalGaussian',)

