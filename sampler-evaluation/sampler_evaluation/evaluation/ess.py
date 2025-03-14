## code to calculate the effective sample size, using ...

import jax
import jax.numpy as jnp
import warnings


def get_num_latents(target):
    return target.ndims


def err(f_true, var_f, contract):
    """Computes the error b^2 = (f - f_true)^2 / var_f
    Args:
        f: E_sampler[f(x)], can be a vector
        f_true: E_true[f(x)]
        var_f: Var_true[f(x)]
        contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max

    Returns:
        contract(b^2)
    """

    return jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))


def samples_to_low_error(err_t, low_error=0.01):
    """Uses the error of the expectation values to compute the effective sample size n_eff
    b^2 = 1/n_eff"""

    cutoff_reached = err_t[-1] < low_error
    # if not cutoff_reached:
    #     jax.debug.print("Error never below threshold, final error is {x}", x=err_t[-1])
    # else:
    #     jax.debug.print("Error below threshold at final error {x}", x=err_t[-1])
    crossing = find_crossing(err_t, low_error)
    return crossing * (1 / cutoff_reached)


# def calculate_ess(err_t, n_eff=100):

#     print("calculating ess")

#     grads_to_low, cutoff_reached = samples_to_low_error(
#         err_t, grad_evals_per_step, 1.0 / n_eff
#     )

#     full_grads_to_low = grads_to_low

#     return (
#         (n_eff / full_grads_to_low) * cutoff_reached,
#         full_grads_to_low * (1 / cutoff_reached),
#         cutoff_reached,
#     )


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    b = array > cutoff
    indices = jnp.argwhere(b, size=array.shape[0])
    if indices.shape[0] == 0:
        warnings.warn("Error always below threshold.")
        return 1

    return jnp.max(indices) + 1


def get_standardized_squared_error(samples, f, E_f, Var_f, contract_fn=jnp.max):
    """
    samples: jnp.array of shape (batch_size, num_samples, dim)
    f: broadcastable function (like lambda x: x**2) that takes in a number and returns a number
    E_f_x: the expected value of f(x) for the distribution of x
    E_f_x2: the expected value of f(x)^2 for the distribution of x
    cost_per_step: the cost of drawing each sample

    returns:
      (E_hat[f(x)] - E[f(x)])^2 / Var[f(x)], where E_hat[f(x)] is the empirical average of f(x) over the samples, with a median across chains, and taking the worst case across dimensions of f(x) is multidimensional
    """
    exps = (
        jnp.cumsum(f(samples), axis=1)
        / jnp.arange(1, samples.shape[1] + 1)[None, :, None]
    )

    error_function = lambda x: contract_fn(jnp.square(x - E_f) / Var_f)

    errors = jnp.median(jax.vmap(jax.vmap(error_function))(exps), axis=0)

    return errors
