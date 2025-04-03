import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 4
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")

from results.run_benchmarks import run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_evaluation.models.banana_mams_paper import banana_mams_paper
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_evaluation.models.phi4 import phi4
from sampler_comparison.samplers.annealing.annealing import annealed
import jax.numpy as jnp 
import numpy as np


identity = lambda x:x

beta_schedule = [10.0, 5.0, 2.0]



# reduced_lam = jnp.linspace(-2.5, 7.5, 8) #lambda range around the critical point (m^2 = -4 is fixed)
reduced_lam = jnp.array([7.5])

def unreduce_lam(reduced_lam, side):
    """see Fig 3 in https://arxiv.org/pdf/2207.00283.pdf"""
    return 4.25 * (reduced_lam * np.power(side, -1.0) + 1.0)

for L in [64]:
    for integrator_type in ['velocity_verlet', 'mclachlan', 'omelyan']:

        lams = unreduce_lam(reduced_lam=reduced_lam,side=L)


        for lam in lams:

            model = phi4(L=L, lam=lam)

            run_benchmarks(
                    models={
                        model.name: model,
                    },
                    samplers={

                        
                        f"adjusted_microcanonical_{integrator_type}": lambda: annealed(adjusted_mclmc, beta_schedule=beta_schedule, intermediate_num_steps=1000, return_only_final=False, kwargs={"num_tuning_steps":5000, "integrator_type":integrator_type}),
                        # f"nuts_{integrator_type}": lambda: annealed(nuts, beta_schedule=beta_schedule, intermediate_num_steps=1000, return_only_final=False,  kwargs={"num_tuning_steps":1000, 'integrator_type':integrator_type, }),
                        # f"unadjusted_microcanonical_{integrator_type}": lambda: annealed(unadjusted_mclmc, beta_schedule=beta_schedule, intermediate_num_steps=10000, return_only_final=False, kwargs={"num_tuning_steps":10000, 'integrator_type':integrator_type}),
                    },
                    batch_size=batch_size,
                    num_steps=50000,
                    save_dir=f"results/Phi4/results",
                    key=jax.random.key(20),
                    map=jax.pmap
                )
