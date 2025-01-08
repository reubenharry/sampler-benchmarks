## code to calculate the effective sample size, using ...

import jax
import jax.numpy as jnp


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


def grads_to_low_error(err_t, grad_evals_per_step=1, low_error=0.01):
    """Uses the error of the expectation values to compute the effective sample size neff
    b^2 = 1/neff"""

    cutoff_reached = err_t[-1] < low_error
    crossing = find_crossing(err_t, low_error) * grad_evals_per_step
    return crossing, cutoff_reached


def calculate_ess(err_t, grad_evals_per_step, num_tuning_steps, neff=100):

    grads_to_low, cutoff_reached = grads_to_low_error(
        err_t, grad_evals_per_step, 1.0 / neff
    )
    # print("grads_to_low", grads_to_low.shape)

    full_grads_to_low = grads_to_low
    # + num_tuning_steps * grad_evals_per_step

    return (
        (neff / full_grads_to_low) * cutoff_reached,
        full_grads_to_low * (1 / cutoff_reached),
        cutoff_reached,
    )


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    b = array > cutoff
    indices = jnp.argwhere(b)
    if indices.shape[0] == 0:
        print("\n\n\nNO CROSSING FOUND!!!\n\n\n", array, cutoff)
        return 1

    return jnp.max(indices) + 1


def sampler_grads_to_low_error(sampler, model, num_steps, batch_size, key, pvmap=jax.vmap):

    try:
        model.sample_transformations[
            "square"
        ].ground_truth_mean, model.sample_transformations[
            "square"
        ].ground_truth_standard_deviation
    except:
        raise AttributeError("Model must have E_x2 and Var_x2 attributes")

    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch_size)

    expectation, metadata = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=num_steps, initial_position=pos, key=key
        )
    )(
        jnp.ones(
            (
                batch_size,
                model.ndims,
            )
        ),
        keys,
    )

    err_t_mean_avg = jnp.median(expectation[:, :, 0], axis=0)
    esses_avg, grads_to_low_avg, _ = calculate_ess(
        err_t_mean_avg,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
        num_tuning_steps=metadata["num_tuning_steps"].mean(),
    )

    err_t_mean_max = jnp.median(expectation[:, :, 1], axis=0)
    esses_max, grads_to_low_max, _ = calculate_ess(
        err_t_mean_max,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
        num_tuning_steps=metadata["num_tuning_steps"].mean(),
    )

    return (
        err_t_mean_max,
        grads_to_low_max,
        err_t_mean_avg,
        grads_to_low_avg,
        expectation,
    )
