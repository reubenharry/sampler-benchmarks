from functools import partial
import itertools
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")  
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))

from results.run_benchmarks import lookup_results, run_benchmarks
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import sampler_evaluation
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_evaluation.models.brownian import brownian_motion
import jax.numpy as jnp
import blackjax
from sampler_comparison.samplers.general import (
    with_only_statistics,
    make_log_density_fn,
)
import blackjax.mcmc.metrics as metrics


from results.run_benchmarks import run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers
from sampler_evaluation.models.brownian import brownian_motion

model = brownian_motion()

# run_benchmarks(
#         models={model.name: model},
#         samplers={
#             # "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),
#             "nuts": partial(nuts,num_tuning_steps=500),
#             # "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),
#             # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
#         },
#         batch_size=batch_size,
#         num_steps=10000,
#         save_dir=f"results/{model.name}",
#         key=jax.random.key(20),
#         map=jax.pmap,
#         calculate_ess_corr=False,
#     )

# run_benchmarks(
#         models={model.name: model},
#         samplers={
#             "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=3e-4, num_tuning_steps=20000, diagonal_preconditioning=True),
#             "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
#         },
#         batch_size=batch_size,
#         num_steps=150000,
#         save_dir=f"results/{model.name}",
#         key=jax.random.key(20),
#         map=jax.pmap,
#         calculate_ess_corr=False,
#     )

mh_options = [True, False]
canonical_options = [True, False]
langevin_options = [True, False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet', 'mclachlan'] # , 'omelyan']
diagonal_preconditioning_options = [True, False]
models = [model]

redo = True 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=100000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)


