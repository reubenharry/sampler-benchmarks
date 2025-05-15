from functools import partial
import os
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
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

from sampler_evaluation.models.neals_funnel_mams_paper import neals_funnel_mams_paper
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import unadjusted_mclmc



run_benchmarks(
        models={
            "Neals_Funnel_MAMS_Paper": neals_funnel_mams_paper,
        },
        samplers={

            "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=40000, warmup='unadjusted_mclmc', target_acc_rate=0.99, num_windows=3),
            "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=40000, warmup='unadjusted_mclmc', target_acc_rate=0.99, num_windows=3),
            "nuts": partial(nuts,num_tuning_steps=5000, target_acc_rate=0.99),
            "unadjusted_microcanonical": unadjusted_mclmc(num_tuning_steps=20000, desired_energy_var=1e-6),
        },
        batch_size=batch_size,
        num_steps=20000,
        save_dir="results/Neals_Funnel_MAMS_Paper",
        key=jax.random.key(19),
        map=jax.pmap
    )
