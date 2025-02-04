def sampler_grads_to_low_error(
    sampler, model, num_steps, batch_size, key, pvmap=jax.vmap
):

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

    squared_errors, metadata = pvmap(
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
    # TODO: propoer initialization!

    err_t_avg_x2 = jnp.median(squared_errors[:, :, 0], axis=0)
    _, grads_to_low_avg_x2, _ = calculate_ess(
        err_t_avg_x2,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_max_x2 = jnp.median(squared_errors[:, :, 1], axis=0)
    _, grads_to_low_max_x2, _ = calculate_ess(
        err_t_max_x2,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_avg_x = jnp.median(squared_errors[:, :, 2], axis=0)
    _, grads_to_low_avg_x, _ = calculate_ess(
        err_t_avg_x,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    err_t_max_x = jnp.median(squared_errors[:, :, 3], axis=0)
    _, grads_to_low_max_x, _ = calculate_ess(
        err_t_max_x,
        grad_evals_per_step=metadata["num_grads_per_proposal"].mean(),
    )

    return (
        {
            "max_over_parameters": {
                "square": {
                    "error": err_t_max_x2,
                    "grads_to_low_error": grads_to_low_max_x2.item(),
                },
                "identity": {
                    "error": err_t_max_x,
                    "grads_to_low_error": grads_to_low_max_x.item(),
                },
            },
            "avg_over_parameters": {
                "square": {
                    "error": err_t_avg_x2,
                    "grads_to_low_error": grads_to_low_avg_x2.item(),
                },
                "identity": {
                    "error": err_t_avg_x,
                    "grads_to_low_error": grads_to_low_avg_x.item(),
                },
            },
            "num_tuning_grads": metadata["num_tuning_grads"].mean().item(),
            "L": metadata["L"].mean().item(),
            "step_size": metadata["step_size"].mean().item(),
        },
        squared_errors,
    )
