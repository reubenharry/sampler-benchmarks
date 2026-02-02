import functools
import numpy as np
from blackjax.adaptation.laps import laps as run_laps
import jax.scipy.stats as stats
import jax.numpy as jnp
import jax

def regression_logprob(log_scale, coefs, preds, x):
        """Linear regression"""
        scale = jnp.exp(log_scale)
        scale_prior = stats.expon.logpdf(scale, 0, 1) + log_scale
        coefs_prior = stats.norm.logpdf(coefs, 0, 5)
        y = jnp.dot(x, coefs)
        logpdf = stats.norm.logpdf(preds, y, scale)
        # reduce sum otherwise broacasting will make the logprob biased.
        return sum(x.sum() for x in [scale_prior, coefs_prior, logpdf])

def test_laps(i):
        """Test the LAPS kernel."""
        init_key0, init_key1, inference_key = jax.random.split(jax.random.key(i), 3)
        x_data = jax.random.normal(init_key0, shape=(1000, 1))
        y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

        logposterior_fn_ = functools.partial(
            regression_logprob, x=x_data, preds=y_data
        )
        # LAPS expects a function that takes an array, not a dictionary
        logposterior_fn = lambda x: logposterior_fn_(log_scale=x[0], coefs=x[1])

        info, grads_per_step, _acc_prob, final_state = run_laps(
            logdensity_fn=logposterior_fn,
            sample_init=lambda key: jax.random.normal(key, shape=(2,)),
            ndims=2,
            num_steps1=1000,
            num_steps2=1000,
            num_chains=100,
            mesh=jax.sharding.Mesh(jax.devices()[:1], "chains"),
            rng_key=jax.random.key(0),
            early_stop=False,
            diagonal_preconditioning=True,
            integrator_coefficients=None,
            steps_per_sample=15,
            r_end=0.01,
            diagnostics=True,
            superchain_size=1,
        )

        scale_samples = np.exp(final_state.position[:, 0])
        coefs_samples = final_state.position[:, 1]

        print(np.mean(scale_samples))
        print(np.mean(coefs_samples))

        np.testing.assert_allclose(np.mean(scale_samples), 1.0, atol=1e-1)
        np.testing.assert_allclose(np.mean(coefs_samples), 3.0, atol=1e-1)


if __name__ == "__main__":
    # i = 0
    # print(f"Code works with i={i}")
    # test_laps(i)
    i = 19
    print(f"Code fails with i={i}")
    test_laps(i)