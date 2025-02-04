import numpy as np
import tensorflow.compat.v2 as tf

# import tensorflow_probability as tfp
import tensorflow_probability.substrates.jax as tfp
from inference_gym.targets import model
import jax.numpy as jnp

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    "Gaussian",
]


class Gaussian(model.Model):
    """Creates a random ill-conditioned Gaussian.

    The covariance matrix has eigenvalues sampled from the inverse Gamma
    distribution with the specified shape, and then rotated by a random orthogonal
    matrix.

    Note that this function produces reproducible targets, i.e. the constructor
    `seed` argument always needs to be non-`None`.
    """

    def __init__(
        self,
        ndims,
        name="gaussian",
        pretty_name="Gaussian",
    ):
        """Construct a Standard multivariate Gaussian.

        Args:
          ndims: Python `int`. Dimensionality of the Gaussian.
          name: Python `str` name prefixed to Ops created by this class.
          pretty_name: A Python `str`. The pretty name of this model.
        """

        self.ndims = ndims

        eigenvalues = jnp.ones(ndims)
        covariance = jnp.diag(eigenvalues)

        gaussian = tfd.MultivariateNormalTriL(
            loc=jnp.zeros(ndims), scale_tril=jnp.linalg.cholesky(covariance)
        )
        # tf.convert_to_tensor(covariance, dtype=tf.float32)))
        self._eigenvalues = eigenvalues

        sample_transformations = {
            "identity": model.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name="Identity",
                ground_truth_mean=np.zeros(ndims),
                ground_truth_standard_deviation=np.sqrt(
                    np.diag(covariance)
                ),  # todo: fix
            ),
            "square": model.Model.SampleTransformation(
                fn=lambda params: params**2,
                pretty_name="Square",
                ground_truth_mean=eigenvalues,
                ground_truth_standard_deviation=np.sqrt(2 * jnp.square(eigenvalues)),
            ),
        }

        self._gaussian = gaussian

        super(Gaussian, self).__init__(
            default_event_space_bijector=tfb.Identity(),
            event_shape=gaussian.event_shape,
            dtype=gaussian.dtype,
            name=name + "_" + str(self.ndims),
            pretty_name=pretty_name,
            sample_transformations=sample_transformations,
        )

    def _unnormalized_log_prob(self, value):
        return self._gaussian.log_prob(value)

    @property
    def covariance_eigenvalues(self):
        return self._eigenvalues

    def sample(self, sample_shape=(), seed=None, name="sample"):
        """Generate samples of the specified shape from the target distribution.

        The returned samples are exact (and independent) samples from the target
        distribution of this model.

        Note that a call to `sample()` without arguments will generate a single
        sample.

        Args:
          sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
          seed: Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.
          name: Name to give to the prefix the generated ops.

        Returns:
          samples: a `Tensor` with prepended dimensions `sample_shape`.
        """
        return self._gaussian.sample(sample_shape, seed=seed, name=name)
