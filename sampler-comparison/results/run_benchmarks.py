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
    models, samplers, batch_size, num_steps, key=jax.random.PRNGKey(1), save_dir=None,map=jax.pmap, calculate_ess_corr=False,
):


    for i, (sampler, model) in enumerate(itertools.product(samplers, models)):
        results = []

        key = jax.random.fold_in(key, i)


        (stats, _) = sampler_grads_to_low_error(
            sampler=map(
            lambda key, pos: samplers[sampler](return_samples=calculate_ess_corr)(
                model=models[model], 
                initial_position=pos, 
                key=key,
                num_steps=num_steps,
                
                )
            ),
            model=models[model],
            batch_size=batch_size,
            key=key,
            calculate_ess_corr=calculate_ess_corr
            # pvmap=map,
        )

        jax.debug.print("stats {x}", x=stats)
        # raise Exception

        for trans in models[model].sample_transformations:

            results.append(
                {
                    "Sampler": sampler,
                    "Model": model,
                    "num_grads_to_low_error": stats["max_over_parameters"][trans][
                        "grads_to_low_error"
                    ],
                    "ess_corr": stats["max_over_parameters"][trans]["autocorrelation"],
                    "max": True,
                    "statistic": trans,
                    "num_tuning_grads": stats["num_tuning_grads"],
                    "L": stats["L"],
                    "step_size": stats["step_size"],
                }
            )
            results.append(
                {
                    "Sampler": sampler,
                    "Model": model,
                    "num_grads_to_low_error": stats["avg_over_parameters"][trans][
                        "grads_to_low_error"
                    ],
                    "ess_corr": stats["avg_over_parameters"][trans]["autocorrelation"],
                    "max": False,
                    "statistic": trans,
                    "num_tuning_grads": stats["num_tuning_grads"],
                    "L": stats["L"],
                    "step_size": stats["step_size"],
                }
            )

        # results.append(
        #     {
        #         "Sampler": sampler,
        #         "Model": model,
        #         "num_grads_to_low_error": stats["max_over_parameters"]["identity"][
        #             "grads_to_low_error"
        #         ],
        #         "ess_corr": stats["max_over_parameters"]["identity"]["autocorrelation"],
        #         "max": True,
        #         "statistic": "x",
        #         "num_tuning_grads": stats["num_tuning_grads"],
        #         "L": stats["L"],
        #         "step_size": stats["step_size"],
        #     }
        # )
        # results.append(
        #     {
        #         "Sampler": sampler,
        #         "Model": model,
        #         "num_grads_to_low_error": stats["avg_over_parameters"]["identity"][
        #             "grads_to_low_error"
        #         ],
        #         "ess_corr": stats["avg_over_parameters"]["identity"]["autocorrelation"],
        #         "max": False,
        #         "statistic": "x",
        #         "num_tuning_grads": stats["num_tuning_grads"],
        #         "L": stats["L"],
        #         "step_size": stats["step_size"],
        #     }
        # )
        df = pd.DataFrame(results)

        if save_dir is not None:
            print(f"Saving results to", os.path.join(save_dir, f"{sampler}_{model}.csv"))
            df.to_csv(os.path.join(save_dir, f"{sampler}_{model}.csv"))


if __name__ == "__main__":
    run_benchmarks(models=models, samplers=samplers, batch_size=128, num_steps=10000)
