import os 
# print(os.listdir("../../blackjax/blackjax"))
# print(os.listdir("../../blackjax"))
import sys
# sys.path.append("../sampler-comparison/src/inference-gym/spinoffs/inference_gym") 
sys.path.append(".")  
# sys.path.append("../../blackjax/blackjax")
sys.path.append("../../blackjax")
sys.path.append("../../blackjax-benchmarks/sampler-comparison")
sys.path.append("../../blackjax-benchmarks/sampler-evaluation")
from results.run_benchmarks import lookup_results
import jax
batch_size = 1
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import itertools
# import pandas as pd
from sampler_evaluation.models.u1 import U1


# run_benchmarks(
#         models={
#             "U1": U1(Lt=8, Lx=8, beta=1),
#         },
#         samplers={

#             "adjusted_microcanonical": partial(adjusted_mclmc,num_tuning_steps=1000),
#             # "adjusted_microcanonical_langevin": partial(adjusted_mclmc,L_proposal_factor=5.0, random_trajectory_length=True, L_factor_stage_3=0.23, num_tuning_steps=5000),
#             "nuts": partial(nuts,num_tuning_steps=1000),
#             # "nuts": partial(annealed(nuts, beta_schedule=[10.0, 5.0, 2.0],intermediate_num_steps=1000),num_tuning_steps=500),
#             "unadjusted_microcanonical": partial(unadjusted_mclmc,num_tuning_steps=10000),
#         },
#         batch_size=batch_size,
#         num_steps=2000,
#         save_dir="results/U1",
#         key=jax.random.key(20),
#         map=jax.pmap
#     )



mh_options = [True]
canonical_options = [True]
langevin_options = [False]
tuning_options = ['nuts']
integrator_type_options = ['velocity_verlet'] # , 'mclachlan', 'omelyan']
diagonal_preconditioning_options = [True]
models = [U1(Lt=16, Lx=16, beta=6)]

redo = True 

for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
    results = lookup_results(model=model, mh=mh, batch_size=batch_size, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo)
    print(results)
