import numpy as np
import tensorflow.compat.v2 as tf

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


# This is (by default) unrotated (unlike the inference-gym version) and is used for a result in a paper.
class IllConditionedGaussian(model.Model):
    def __init__(
        self,
        ndims,
        condition_number,
        eigenvalues="linear",
        key=None,
        name="ICG",
        pretty_name="Ill_Conditioned_Gaussian",
        do_covariance=True,
    ):
        self.ndims = ndims
        self.condition_number = condition_number

        # fix the eigenvalues of the covariance matrix
        if eigenvalues == "linear":
            eigs = jnp.linspace(1.0 / condition_number, 1, ndims)
        elif eigenvalues == "log":
            eigs = jnp.logspace(
                -0.5 * jnp.log10(condition_number),
                0.5 * jnp.log10(condition_number),
                ndims,
            )
        elif eigenvalues == "outliers":
            num_outliers = 2
            eigs = jnp.concatenate(
                (
                    jnp.ones(num_outliers) * condition_number,
                    jnp.ones(ndims - num_outliers),
                )
            )
        else:
            raise ValueError(
                "eigenvalues = " + str(eigenvalues) + " is not a valid option."
            )

        if key == None:  # diagonal covariance matrix
            self.e_x2 = eigs
            # self.R = jnp.eye(ndims)
            self.inv_cov = 1.0 / eigs
            if do_covariance:
                self.cov = jnp.diag(eigs)

            self._unnormalized_log_prob = lambda x: -0.5 * jnp.sum(
                jnp.square(x) * self.inv_cov
            )

        else:  # randomly rotate
            D = jnp.diag(eigs)
            inv_D = jnp.diag(1 / eigs)
            R, _ = jnp.array(
                jnp.linalg.qr(jax.random.normal(key=key, shape=((ndims, ndims))))
            )  # random rotation
            self.R = R
            self.inv_cov = R @ inv_D @ R.T
            self.cov = R @ D @ R.T
            self.e_x2 = jnp.diagonal(R @ D @ R.T)
            self._unnormalized_log_prob = lambda x: -0.5 * x.T @ self.inv_cov @ x

        self.e_x = jnp.zeros(ndims)
        self.var_x2 = 2 * jnp.square(self.e_x2)

        # self.exact_sample = lambda key: self.R @ (
        #     jax.random.normal(key, shape=(ndims,)) * jnp.sqrt(eigs)
        # )

        sample_transformations = {
            "identity": model.Model.SampleTransformation(
                fn=lambda params: params,
                pretty_name="Identity",
                ground_truth_mean=self.e_x,
                ground_truth_standard_deviation=jnp.sqrt(self.e_x2 - self.e_x**2),
            ),
            "square": model.Model.SampleTransformation(
                fn=lambda params: params**2,
                pretty_name="Square",
                ground_truth_mean=self.e_x2,
                ground_truth_standard_deviation=jnp.sqrt(self.var_x2),
            ),
        }
        if do_covariance:

            sample_transformations["covariance"] = model.Model.SampleTransformation(
                fn=lambda params: jnp.outer(params - self.e_x, params - self.e_x),
                pretty_name="Covariance",
                ground_truth_mean=self.cov,
                ground_truth_standard_deviation=jnp.nan,
            )
        super(IllConditionedGaussian, self).__init__(
            default_event_space_bijector=tfb.Identity(),
            event_shape=tf.TensorShape([ndims]),
            dtype=np.float32,
            name=name+f"_{self.ndims}_{self.condition_number}",
            pretty_name=pretty_name,
            sample_transformations=sample_transformations,
        )
