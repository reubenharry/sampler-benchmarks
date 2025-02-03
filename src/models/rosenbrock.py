import numpy as np
import tensorflow.compat.v2 as tf

# import tensorflow_probability as tfp
import tensorflow_probability.substrates.jax as tfp
from inference_gym.targets import model
import jax.numpy as jnp

tfb = tfp.bijectors
tfd = tfp.distributions
import inference_gym.using_jax as gym
from inference_gym.targets import model
import jax.numpy as jnp
import jax


class Rosenbrock_36D(model.Model):
    def __init__(
        self,
        name="gaussian",
        pretty_name="Gaussian",
    ):
        ndims = 36
        Q = 0.1
        D = ndims // 2
        E_x = jnp.array(
            [
                1.0,
            ]
            * D
            + [
                2.0,
            ]
            * D
        )
        E_x2 = jnp.array(
            [
                2.0,
            ]
            * D
            + [
                10.10017429,
            ]
            * D
        )
        Var_x2 = jnp.array(
            [
                6.00036273,
            ]
            * D
            + [
                668.69693635,
            ]
            * D
        )

        sample_transformations = {
            "identity": model.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name="Identity",
                ground_truth_mean=E_x,
                ground_truth_standard_deviation=jnp.sqrt(E_x2 - E_x**2),
            ),
            "square": model.Model.SampleTransformation(
                fn=lambda params: params**2,
                pretty_name="Square",
                ground_truth_mean=E_x2,
                ground_truth_standard_deviation=jnp.sqrt(Var_x2),
            ),
        }

        super(Rosenbrock_36D, self).__init__(
            default_event_space_bijector=tfb.Identity(),
            event_shape=TODO,
            dtype=TODO,
            name=name + "_" + str(self.ndims),
            pretty_name=pretty_name,
            sample_transformations=sample_transformations,
        )

    def logdensity_fn(self, x):
        """- log p of the target distribution"""
        X, Y = x[..., : self.ndims // 2], x[..., self.ndims // 2 :]
        return -0.5 * jnp.sum(
            jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / self.Q, axis=-1
        )

    def sample_init(self, key):
        return jax.random.normal(key, shape=(self.ndims,))
