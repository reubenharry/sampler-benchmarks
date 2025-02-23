import numpy as np
import tensorflow.compat.v2 as tf

# import tensorflow_probability as tfp
import tensorflow_probability.substrates.jax as tfp
from inference_gym.targets import model
import jax.numpy as jnp

tfb = tfp.bijectors
tfd = tfp.distributions
from inference_gym.targets import model
import jax.numpy as jnp
import jax
import pickle


class Rosenbrock_36D(model.Model):
    def __init__(
        self,
        name="rosenbrock_36D",
        pretty_name="Rosenbrock_36D",
    ):
        ndims = 36
        self.ndims = ndims
        self.Q = 0.1

        # todo: ground truths should be calculated from 2D rosenbrock and then composed as independent products
        # dirr = "/global/homes/r/reubenh/blackjax-benchmarks"

        # with open(
        #     f"{dirr}/sampler-evaluation/sampler_evaluation/models/data/rosenbrock_36D_expectations.pkl",
        #     "rb",
        # ) as f:
        #     stats = pickle.load(f)

        # e_x = stats["e_x"]
        # e_x2 = stats["e_x2"]
        # e_x4 = stats["e_x4"]
        # var_x2 = e_x4 - e_x2**2

        # print(e_x.shape, e_x2.shape, e_x4.shape, var_x2.shape)

        # raise Exception

        D = ndims // 2
        e_x = jnp.array(
            [
                1.0,
            ]
            * D
            + [
                2.0,
            ]
            * D
        )
        e_x2 = jnp.array(
            [
                2.0,
            ]
            * D
            + [
                10.10017429,
            ]
            * D
        )
        var_x2 = jnp.array(
            [
                6.00036273,
            ]
            * D
            + [
                668.69693635,
            ]
            * D
        )

        # print(e_x.shape, e_x2.shape, var_x2.shape)

        # raise Exception

        sample_transformations = {
            "identity": model.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name="Identity",
                ground_truth_mean=e_x,
                ground_truth_standard_deviation=jnp.sqrt(e_x2 - e_x**2),
            ),
            "square": model.Model.SampleTransformation(
                fn=lambda params: params**2,
                pretty_name="Square",
                ground_truth_mean=e_x2,
                ground_truth_standard_deviation=jnp.sqrt(var_x2),
            ),
        }

        super(Rosenbrock_36D, self).__init__(
            default_event_space_bijector=tfb.Identity(),
            event_shape=tf.TensorShape([ndims]),
            dtype=np.float32,
            name=name,
            pretty_name=pretty_name,
            sample_transformations=sample_transformations,
        )

    def unnormalized_log_prob(self, x):
        """- log p of the target distribution"""
        X, Y = x[..., : self.ndims // 2], x[..., self.ndims // 2 :]
        return -0.5 * jnp.sum(
            jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / self.Q, axis=-1
        )

    def exact_sample(self, key):
        x = jax.random.normal(key=key, shape=(self.ndims // 2,)) + 1.0
        y = jax.random.normal(key=key, shape=(self.ndims // 2,)) * jnp.sqrt(
            self.Q
        ) + jnp.square(x)
        return jnp.array([x, y]).reshape((self.ndims,))
