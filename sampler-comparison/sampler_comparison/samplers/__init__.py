from functools import partial
from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import adjusted_mclmc
from sampler_comparison.samplers.microcanonicalmontecarlo.unadjusted import (
    unadjusted_mclmc,
)
from sampler_comparison.samplers.hamiltonianmontecarlo.unadjusted.underdamped_langevin import unadjusted_lmc
from sampler_comparison.samplers.hamiltonianmontecarlo.hmc import adjusted_hmc


samplers={

            "adjusted_hmc": partial(adjusted_hmc,num_tuning_steps=5000, integrator_type="velocity_verlet"),

            "nuts": partial(nuts,num_tuning_steps=5000),

            "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=5000),

            # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),

            "underdamped_langevin": partial(unadjusted_lmc,desired_energy_var=1e-4, num_tuning_steps=20000, diagonal_preconditioning=True),

            "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=20000),
        }
