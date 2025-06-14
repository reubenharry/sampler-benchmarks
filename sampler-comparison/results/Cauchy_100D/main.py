from functools import partial
import itertools
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 64
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

import sys
sys.path.append(".")
import sys
sys.path.append(".")  
sys.path.append("../../blackjax")
sys.path.append("../../sampler-benchmarks/sampler-comparison")
sys.path.append("../../sampler-benchmarks/sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
print(os.listdir("../../src/inference-gym/spinoffs/inference_gym"))


from results.run_benchmarks import lookup_results, run_benchmarks
import sampler_evaluation
from sampler_comparison.samplers import samplers
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc, unadjusted_lmc_no_tuning
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
from sampler_evaluation.models.banana import banana
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import blackjax
import jax.numpy as jnp
from sampler_comparison.samplers.general import (
    make_log_density_fn,
)
from sampler_comparison.util import (
    calls_per_integrator_step,
    map_integrator_type_to_integrator,
)
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from sampler_evaluation.models.german_credit import german_credit
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc

from sampler_evaluation.models.item_response import item_response

model = item_response()

mh_options = [True]
canonical_options = [True, False]
langevin_options = [True, False]
tuning_options = ['alba']
integrator_type_options = ['velocity_verlet', 'mclachlan'] # , 'omelyan']
diagonal_preconditioning_options = [True, False]
models = [model]

redo = False 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=20000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)

mh_options = [False]

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, num_steps=200000, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)


# from functools import partial
# import os
# import jax
# jax.config.update("jax_enable_x64", True)

# batch_size = 128
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
# num_cores = jax.local_device_count()

# import sys
# sys.path.append(".")

# from results.run_benchmarks import run_benchmarks
# from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
#     adjusted_mclmc,
# )
# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
# import sampler_evaluation
# from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
# from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc
# from sampler_comparison.samplers.grid_search.grid_search import grid_search_hmc
# from sampler_comparison.samplers import samplers
# from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
# from sampler_evaluation.models.item_response import item_response

# model = item_response()

# run_benchmarks(
#         models={model.name: model},
#         samplers={
#             "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),
#             "nuts": partial(nuts,num_tuning_steps=500),
#             "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),
#             "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
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
#         num_steps=40000,
#         save_dir=f"results/{model.name}",
#         key=jax.random.key(20),
#         map=jax.pmap,
#         calculate_ess_corr=False,
#     )
# from functools import partial
# import os
# import jax
# jax.config.update("jax_enable_x64", True)

# batch_size = 64
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
# num_cores = jax.local_device_count()

# import sys
# sys.path.append(".")

# from results.run_benchmarks import run_benchmarks
# from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
#     adjusted_mclmc,
# )
# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
# import sampler_evaluation
# from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
# from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc
# from sampler_evaluation.models.cauchy import cauchy
# from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
# from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc

# samplers={

#             "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=500, diagonal_preconditioning=False),
#             "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000, diagonal_preconditioning=False),
#             "nuts": partial(nuts,num_tuning_steps=500, diagonal_preconditioning=False),


#             "adjusted_malt": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet", L_proposal_factor=1.25),
#             "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-4, num_tuning_steps=30000, diagonal_preconditioning=True),
#             "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
#         }

# ndims = 100

# model = cauchy(ndims=ndims)

# run_benchmarks(
#         models={model.name: model},
#         samplers=samplers,
#         batch_size=batch_size,
#         num_steps=200000,
#         save_dir=f"results/{model.name}",
#         key=jax.random.key(19),
#         map=jax.pmap
#     )
