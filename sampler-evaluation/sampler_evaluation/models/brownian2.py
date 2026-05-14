import numpy as np
import tensorflow.compat.v2 as tf
import os

# import tensorflow_probability as tfp
import tensorflow_probability.substrates.jax as tfp
# import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym")
from inference_gym.targets import model
import jax.numpy as jnp

tfb = tfp.bijectors
tfd = tfp.distributions
from inference_gym.targets import model
import jax.numpy as jnp
import jax
import pickle

module_dir = os.path.dirname(os.path.abspath(__file__))



class Brownian(model.Model):

    def __init__(self):

        name = 'vector_brownian_motion_unknown_scales_missing_middle_observations'

        with open(
            # f"{module_dir}/data/{brownian_motion.name}_expectations_old.pkl",
            f"{module_dir}/data/{name}_expectations.pkl",
            "rb",
        ) as f:
            stats = pickle.load(f)

        # e_x = stats["e_x"]
        # e_x2 = stats["e_x2"]
        # e_x4 = stats["e_x4"]
        e_x = stats["identity"]
        e_x2 = stats["square"]
        e_x4 = stats["quartic"]
        cov = stats["covariance"]
        # import jax
        # jax.debug.print("cov {x}", x=jnp.any(jnp.isnan(cov)))
        # raise Exception("stop")
        var_x2 = e_x4 - e_x2**2


        sample_transformations = {
            # "identity": model.Model.SampleTransformation(
            #     fn=lambda params: gym.targets.VectorModel(
            #         gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations(),
            #         flatten_sample_transformations=True, # TODO: sub out
            #     ).sample_transformations["identity"](params),
            #     pretty_name="Identity",
            #     ground_truth_mean=e_x,
            #     ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),
            # ),
            "square": model.Model.SampleTransformation(
            fn=lambda params: self.sample_transformations["identity"](params)
            ** 2,
            pretty_name="Square",
            ground_truth_mean=e_x2,
            ground_truth_standard_deviation=jnp.sqrt(var_x2),
            ),
            "quartic": model.Model.SampleTransformation(
            fn=lambda params: self.sample_transformations["identity"](params)
            ** 4,
            pretty_name="Quartic",
            ground_truth_mean=e_x4,
            ground_truth_standard_deviation=jnp.nan,
            ),
            "covariance": model.Model.SampleTransformation(
                fn=lambda params: jnp.outer(params - e_x, params - e_x),
                pretty_name="Covariance",
                ground_truth_mean=cov,
                ground_truth_standard_deviation=jnp.nan,
            )
        }
        


        num_data = 30
        self.ndims = num_data + 2
        
        data = jnp.array([0.21592641, 0.118771404, -0.07945447, 0.037677474, -0.27885845, -0.1484156, -0.3250906, -0.22957903,
                               -0.44110894, -0.09830782, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8786016, -0.83736074,
                               -0.7384849, -0.8939254, -0.7774566, -0.70238715, -0.87771565, -0.51853573, -0.6948214, -0.6202789])
        # sigma_obs = 0.15, sigma_i = 0.1

        observable = jnp.concatenate((jnp.ones(10), jnp.zeros(10), jnp.ones(10)))
        num_observable = jnp.sum(observable)  # = 20


        def logp(x):
            # y = softplus_to_log(x[:2])

            lik = 0.5 * jnp.exp(-2 * x[1]) * jnp.sum(observable * jnp.square(x[2:] - data)) + x[1] * num_observable
            prior_x = 0.5 * jnp.exp(-2 * x[0]) * (x[2] ** 2 + jnp.sum(jnp.square(x[3:] - x[2:-1]))) + x[0] * num_data
            prior_logsigma = 0.5 * jnp.sum(jnp.square(x / 2.0))

            return -lik - prior_x - prior_logsigma


        def sample_init(key):
            key_walk, key_sigma = jax.random.split(key)

            # original prior
            # log_sigma = jax.random.normal(key_sigma, shape= (2, )) * 2

            # narrower prior

            sigma = jnp.exp(jnp.log(np.array([0.1, 0.15])) + jax.random.normal(key_sigma, shape=(2,)) * 0.1)  # *0.05# log sigma_i, log sigma_obs
            inv_soft_plus = lambda x: jnp.log(jnp.exp(x)-1.)
            sigma_transformed = inv_soft_plus(sigma)

            walk = random_walk(key_walk, self.ndims - 2) * sigma[0]

            return jnp.concatenate((sigma_transformed, walk))
        
    
        self._unnormalized_log_prob = logp
        self.sample_init = sample_init



        super(Brownian, self).__init__(
            default_event_space_bijector=tfb.Identity(),
            event_shape=tf.TensorShape([self.ndims]),
            dtype=np.float32,
            name=name,
            pretty_name='Brownian Motion with Unknown Scales',
            sample_transformations=sample_transformations,
        )




def random_walk(key, num):
    """ Genereting process for the standard normal walk:
        x[0] ~ N(0, 1)
        x[n+1] ~ N(x[n], 1)

        Args:
            key: jax random key
            num: number of points in the walk
        Returns:
            1 realization of the random walk (array of length num)
    """

    def step(track, useless):
        x, key = track
        randkey, subkey = jax.random.split(key)
        x += jax.random.normal(subkey)
        return (x, randkey), x

    return jax.lax.scan(step, init=(0.0, key), xs=None, length=num)[1]