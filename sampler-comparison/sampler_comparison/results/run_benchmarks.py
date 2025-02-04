import os
import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)

num_cores = jax.local_device_count()
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from sampler_comparison.samplers import samplers
from sampler_evaluation.models import models
# from sampler_comparison.evaluation.ess import sampler_grads_to_low_error
import pandas as pd
from sampler_evaluation.evaluation.ess import calculate_ess
import jax.numpy as jnp

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


def run_benchmarks(models, batch_size, num_steps, key=jax.random.PRNGKey(1), save_dir="results"):

    for i, (sampler, model) in enumerate(itertools.product(samplers, models)):
        results = []

        key = jax.random.fold_in(key, i)

        print(f"Running sampler {sampler} on model {model}")

        (stats, _,) = sampler_grads_to_low_error(
            sampler=samplers[sampler](),
            model=models[model],
            num_steps=num_steps,
            batch_size=batch_size,
            key=key,
            pvmap=jax.pmap,
        )

        results.append(
            {
                "Sampler": sampler,
                "Model": model,
                "num_grads_to_low_error": stats["max_over_parameters"]["square"][
                    "grads_to_low_error"
                ],
                "max": True,
                "statistic": "x2",
                "num_tuning_grads": stats["num_tuning_grads"],
                "L": stats["L"],
                "step_size": stats["step_size"],
            }
        )
        results.append(
            {
                "Sampler": sampler,
                "Model": model,
                "num_grads_to_low_error": stats["avg_over_parameters"]["square"][
                    "grads_to_low_error"
                ],
                "max": False,
                "statistic": "x2",
                "num_tuning_grads": stats["num_tuning_grads"],
                "L": stats["L"],
                "step_size": stats["step_size"],
            }
        )

        results.append(
            {
                "Sampler": sampler,
                "Model": model,
                "num_grads_to_low_error": stats["max_over_parameters"]["identity"][
                    "grads_to_low_error"
                ],
                "max": True,
                "statistic": "x",
                "num_tuning_grads": stats["num_tuning_grads"],
                "L": stats["L"],
                "step_size": stats["step_size"],
            }
        )
        results.append(
            {
                "Sampler": sampler,
                "Model": model,
                "num_grads_to_low_error": stats["avg_over_parameters"]["identity"][
                    "grads_to_low_error"
                ],
                "max": False,
                "statistic": "x",
                "num_tuning_grads": stats["num_tuning_grads"],
                "L": stats["L"],
                "step_size": stats["step_size"],
            }
        )
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_dir, f"{sampler}_{model}.csv"))


if __name__ == "__main__":
    run_benchmarks(models=models, batch_size=64, num_steps=10000)
