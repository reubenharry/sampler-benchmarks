"""
This script is used to estimate the expectations of the model using a long NUTS run.
By default the script runs 4 chains, in parallel.
"""

# TODO: # The script should return diagnostics

import pickle
import sys
import numpy as np
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(4)
sys.path.append("./")
sys.path.append("../sampler-comparison")
sys.path.append("../../blackjax")
import jax
import jax.numpy as jnp
import time

# from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_evaluation.models.phi4 import phi4
from sampler_evaluation.models.u1 import U1
from sampler_evaluation.models.neals_funnel import neals_funnel
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_comparison.samplers.annealing.annealing import annealed
from sampler_evaluation.models.stochastic_volatility_artificial_20000 import stochastic_volatility_artificial_20000


def relative_fluctuations(expectation, square_expectation):

    expectation_mean = jnp.mean(expectation, axis=0)
    std = jnp.sqrt(jnp.mean(square_expectation, axis=0) - expectation_mean**2)
    diff = jnp.abs((expectation - expectation_mean[None, :]) / std[None, :])
    return jnp.max(diff)


def expectations_from_exact_samples(model, key, num_samples=1000):
    try:
        sampling_function = model.exact_sample
    except AttributeError:
        raise AttributeError("Model does not have an exact sample function")
    samples = jax.vmap(sampling_function)(jax.random.split(key, num_samples))

    e_x = jnp.mean(samples, axis=0)
    e_x2 = jnp.mean(jnp.square(samples), axis=0)
    e_x4 = jnp.mean(samples**4, axis=0)
    return e_x, e_x2, e_x4


def estimate_ground_truth(model, num_samples, annealing=False):

    if hasattr(model, "exact_sample") and model.exact_sample is not None:
        key = jax.random.PRNGKey(0)
        e_x, e_x2, e_x4 = expectations_from_exact_samples(
            model, key, num_samples=num_samples
        )
        results = {
            "e_x": e_x,
            "e_x2": e_x2,
            "e_x4": e_x4,
        }

    else:

        sampler = nuts(
            integrator_type="velocity_verlet",
            diagonal_preconditioning=True,
            return_samples=False,
            incremental_value_transform=lambda x: x,
            num_tuning_steps=5000,
            return_only_final=True,
        )

        if annealing:
            sampler = annealed(nuts, beta_schedule=[10.0, 5.0, 2.0], intermediate_num_steps=100000, return_only_final=True,kwargs={
                'integrator_type':"velocity_verlet",
                'diagonal_preconditioning':True,
                # 'return_samples':True,
                'incremental_value_transform':lambda x: x,
                'num_tuning_steps':5000,
                # 'return_only_final':False,
            })

        key = jax.random.PRNGKey(1)
        run_keys = jax.random.split(key, num_chains)
        init_pos = jax.random.normal(
            shape=(
                num_chains,
                model.ndims,
            ),
            key=jax.random.key(0),
        )

        expectation, _ = jax.vmap(
            lambda pos, key: sampler(
                model=model,
                num_steps=num_samples,
                initial_position=pos,
                key=key,
            )
        )(init_pos, run_keys)

        expectation = np.array(expectation)

        e_x = expectation[:, 0, :]
        e_x2 = expectation[:, 1, :]
        e_x4 = expectation[:, 2, :]

        e_x_avg = jnp.nanmean(e_x, axis=0)
        e_x2_avg = jnp.nanmean(e_x2, axis=0)
        e_x4_avg = jnp.nanmean(e_x4, axis=0)

        jax.debug.print("shape {x}", x=e_x_avg.shape)

        # jax.debug.print("avgs {x}", x=(e_x_avg, e_x2_avg, e_x4_avg))

        results = {
            "e_x": e_x_avg,
            "e_x2": e_x2_avg,
            "e_x4": e_x4_avg,
            # "potential_scale_reduction": (
            #     potential_scale_reduction(e_x),
            #     potential_scale_reduction(e_x2),
            #     potential_scale_reduction(e_x4),
            # ),
            "relative_fluctuations": (
                relative_fluctuations(e_x, e_x2),
                relative_fluctuations(e_x2, e_x4),
            ),
        }

    # print(results)

    ## pickle to data
    with open(
        f"./sampler_evaluation/models/data/{model.name}_expectations.pkl", "wb"
    ) as f:
        pickle.dump(results, f)

    if not hasattr(model, "exact_sample"):
        assert quality_check(results), f"More samples needed. Stats are now: {results}"


def quality_check(stats):
    # return stats["potential_scale_reduction"][0] < 1.1 and stats["potential_scale_reduction"][1] < 1.1 and
    return (
        stats["relative_fluctuations"][0] < 0.1
        and stats["relative_fluctuations"][1] < 0.1
    )


if __name__ == "__main__":

    num_chains = 4

    # gold_standard_expectation_steps = {
        # phi4(L=4, lam=1): 40000,
        # banana_mams_paper: 10000000,
    # }
        # neals_funnel(): 1000000,
        # Gaussian(ndims=100) : 10000
        # IllConditionedGaussian(ndims=100, condition_number=100, eigenvalues='log') : 10000,
        # sampler_evaluation.models.brownian_motion(): 2000000,
        # sampler_evaluation.models.german_credit(): 10000000,
        # sampler_evaluation.models.stochastic_volatility(): 1000,
        # stochastic_volatility_mams_paper: 400000,
        # sampler_evaluation.models.item_response(): 1000000,
        # Rosenbrock(): 10000000,
    
    reduced_lam = jnp.linspace(-2.5, 7.5, 8)[6:] #lambda range around the critical point (m^2 = -4 is fixed)


    def unreduce_lam(reduced_lam, side):
        """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
        return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)


    for L in [128,256,512,1024]:
        lams = unreduce_lam(reduced_lam=reduced_lam,side=L)
        for lam in lams:
    
            model = phi4(L=L, lam=lam)

            print(f"Estimating ground truth for {model}")
            toc = time.time()
            ### SET ANNEALING TO TRUE!!!
            estimate_ground_truth(model, num_samples=100000, annealing=True)
            tic = time.time()
            print(f"Time taken: {tic - toc}")
            print("Done")

    # # model = U1(Lt=4, Lx=4, beta=1)
    # model = stochastic_volatility_artificial_20000

    # print(f"Estimating ground truth for {model}")
    # toc = time.time()
    # ### SET ANNEALING TO TRUE!!!
    # estimate_ground_truth(model, num_samples=1000000, annealing=False)
    # tic = time.time()
    # print(f"Time taken: {tic - toc}")
    # print("Done")
