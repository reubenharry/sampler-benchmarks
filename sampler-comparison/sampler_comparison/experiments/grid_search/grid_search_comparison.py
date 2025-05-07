import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(64)
import jax
jax.config.update("jax_enable_x64", True)

num_cores = jax.local_device_count()
import sys

sys.path.append("/global/homes/r/reubenh/blackjax-benchmarks/sampler-comparison")
from sampler_comparison.samplers.grid_search.grid_search import grid_search_L
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc_no_tuning,
)
from results.run_benchmarks import run_benchmarks
from sampler_evaluation.models import models
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper


def benchmark_grid_search(model):

    L, step_size, num_grads, num_grads_avg, edge, inverse_mass_matrix, initial_state = (
        grid_search_L(
            model=model,
            num_steps=40000,
            num_chains=64,
            integrator_type="velocity_verlet",
            key=jax.random.key(0),
            grid_size=10,
            grid_iterations=2,
            opt="max",
        )
    )

    run_benchmarks(
        models={model.name: model},
        samplers={
            f"adjusted_microcanonical_gridsearch": lambda: adjusted_mclmc_no_tuning(
                initial_state=initial_state,
                integrator_type="velocity_verlet",
                L=L,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
            ),
        },
        batch_size=64,
        num_steps=40000,
        save_dir="sampler_comparison/results",
    )


if __name__ == "__main__":

    from sampler_evaluation.models.banana import banana
    from sampler_evaluation.models.brownian import brownian_motion
    from sampler_evaluation.models.german_credit import german_credit
    from sampler_evaluation.models.stochastic_volatility import stochastic_volatility
    from sampler_evaluation.models.item_response import item_response
    from sampler_evaluation.models.rosenbrock import Rosenbrock
    from sampler_evaluation.models.neals_funnel import neals_funnel

    models = {
        # "Banana": banana(),
        "Gaussian_2D": IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        # "Gaussian_100D": Gaussian(ndims=100),
        # 'Gaussian_100D': IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log'),
        # "Brownian_Motion": brownian_motion(),
        # "German_Credit": german_credit(),
        # "Rosenbrock": Rosenbrock(),
        # "Neals_Funnel": neals_funnel(),
        # "Stochastic_Volatility": stochastic_volatility(),
        "Stochastic_Volatility_MAMS_Paper": stochastic_volatility_mams_paper,
        # "Item_Response": item_response(),
    }
    for model in models:

        benchmark_grid_search(models[model])
        print(f"Finished benchmarking {model}")

    # benchmark_grid_search(sampler_evaluation.models.brownian_motion())
