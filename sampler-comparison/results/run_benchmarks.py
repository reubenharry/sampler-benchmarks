import os
import jax

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()
import itertools
import sys

sys.path.append("..")
sys.path.append(".")
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.general import sampler_grads_to_low_error
from sampler_evaluation.models import models
import pandas as pd


def run_benchmarks(
    models, samplers, batch_size, num_steps, key=jax.random.PRNGKey(1), save_dir=None,map=jax.pmap
):


    for i, (sampler, model) in enumerate(itertools.product(samplers, models)):
        results = []

        key = jax.random.fold_in(key, i)


        (stats, _) = sampler_grads_to_low_error(
            sampler=map(
            lambda key, pos: samplers[sampler]()(
                model=models[model], num_steps=num_steps, initial_position=pos, key=key
                )
            ),
            model=models[model],
            num_steps=num_steps,
            batch_size=batch_size,
            key=key,
            # pvmap=map,
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

        if save_dir is not None:
            print(f"Saving results to", os.path.join(save_dir, f"{sampler}_{model}.csv"))
            df.to_csv(os.path.join(save_dir, f"{sampler}_{model}.csv"))


if __name__ == "__main__":
    run_benchmarks(models=models, samplers=samplers, batch_size=128, num_steps=10000)
