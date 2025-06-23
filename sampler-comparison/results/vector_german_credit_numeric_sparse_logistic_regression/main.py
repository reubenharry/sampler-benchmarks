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

model = german_credit()


mh_options = [True, False]
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