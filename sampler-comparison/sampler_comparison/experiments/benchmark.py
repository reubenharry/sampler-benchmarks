
# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"       # defrags GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"      # don't grab all VRAM up front
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 128
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

# print(os.listdir("../../../sampler-comparison"))

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable preallocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
import numpy as np

import pandas as pd
import os
import jax
from results.run_benchmarks import lookup_results
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sampler_comparison.experiments.utils import model_info
import time 
import gc
from sampler_evaluation.models.cauchy import cauchy
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_evaluation.models.brownian import brownian_motion
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
from sampler_evaluation.models.item_response import item_response
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.phi4 import phi4
from sampler_evaluation.models.data.estimate_expectations_phi4 import unreduce_lam
from sampler_evaluation.models.u1 import U1
from sampler_evaluation.models.bimodal import bimodal_gaussian
def clear_jax_cache():
    """Clear JAX compilation cache to prevent memory accumulation"""
    try:
        # Clear JAX compilation cache
        jax.clear_caches()
        print("  ✓ Cleared JAX compilation cache")
    except Exception as e:
        print(f"  ⚠ Warning: Could not clear JAX cache: {e}")
    
    # Force garbage collection
    gc.collect()
    print("  ✓ Forced garbage collection")

def run(models, key, mh_options=[True, False], canonical_options=[True, False], langevin_options=[True, False], tuning_options=['alba', 'grid_search'], integrator_type_options=['velocity_verlet','mclachlan'], diagonal_preconditioning_options=[True, False], redo=False, compute_missing=True, redo_bad_results=None):

    full_results = pd.DataFrame()
    total_combinations = len(list(itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models)))
    current_combination = 0
    
    print(f"Running {total_combinations} total combinations...")
    
    for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
        current_combination += 1
        time_start = time.time()
        
        print(f"\n--- Combination {current_combination}/{total_combinations} ---")
        print(f"Model: {model.name}")
        print(f"MH: {mh}, Canonical: {canonical}, Langevin: {langevin}")
        print(f"Tuning: {tuning}, Integrator: {integrator_type}, Diagonal: {diagonal_preconditioning}")
        
        # Get model-specific parameters with fallbacks for models not in model_info
        if model.name in model_info:
            model_num_steps = model_info[model.name]['num_steps'][mh]
            model_batch_size = model_info[model.name]['batch_size']
        else:
            # Fallback values for models not in model_info
            print(f"Warning: Model {model.name} not found in model_info, using fallback values")
            # model_num_steps = (40000)*np.ceil(model.ndims**0.25).astype(int) if mh else 40000  # Default steps based on adjusted/unadjusted
            # model_batch_size = min(4 + 1000 // (model.ndims), batch_size)  # Default batch size
            model_num_steps = 100
            model_batch_size = 1
        
        results = lookup_results(
            model=model, 
            key=key,
            num_steps=model_num_steps, 
            mh=mh, 
            canonical=canonical, 
            langevin=langevin, 
            tuning=tuning, 
            integrator_type=integrator_type, 
            diagonal_preconditioning=diagonal_preconditioning, 
            redo=redo, 
            batch_size=model_batch_size, 
            relative_path='./', 
            compute_missing=compute_missing,
            redo_bad_results=redo_bad_results
        )
        full_results = pd.concat([full_results, results], ignore_index=True)
        time_end = time.time()
        print(f"✓ Completed in {time_end - time_start:.2f} seconds")
        
        # Clear JAX cache after each combination to prevent memory accumulation
        if tuning == 'grid_search':
            print("  Clearing JAX cache after grid search...")
            clear_jax_cache()
   
if __name__ == "__main__":

    

    print("Starting benchmark...")


    # models = [U1(Lt=side, Lx=side, beta=2.) for side in [512, 1024]]
    # models = [phi4(side, unreduce_lam(reduced_lam=4.0, side=side)) for side in [1024]]

    models = [
        brownian_motion(),
        # Rosenbrock(18),
        # phi4(256, unreduce_lam(reduced_lam=4.0, side=256)),
        # phi4(1024, unreduce_lam(reduced_lam=4.0, side=1024)),
        german_credit(),
        # banana(),
        bimodal_gaussian(),
        stochastic_volatility_mams_paper,
        # IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log', do_covariance=False),
        # IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log', do_covariance=False),
        # item_response(),
        # IllConditionedGaussian(ndims=1000000, condition_number=1, eigenvalues='log', do_covariance=False),
        cauchy(ndims=100),
        # U1(Lt=16, Lx=16, beta=6)
        ]

    # print(models[0].sample_transformations["square"].ground_truth_mean.shape, "ground truth mean shape")

    # raise Exception

    # run(
    #             key=jax.random.PRNGKey(4),
    #             models=[models[0]],
    #             tuning_options=['alba'],
    #             mh_options = [False],
    #             canonical_options = [False],
    #             langevin_options = [True],
    #             integrator_type_options = ['velocity_verlet'],
    #             diagonal_preconditioning_options = [False],
    #             redo=True,
    #             compute_missing=True,
    #             # redo_bad_results=True
    #         )

    for model in models:
        run(
                    key=jax.random.PRNGKey(4),
                    models=[model],
                    tuning_options=['mala'],
                    mh_options = [True],
                    canonical_options = [True],
                    langevin_options = [False],
                    integrator_type_options = ['velocity_verlet'],
                    diagonal_preconditioning_options = [True],
                    redo=True,
                    compute_missing=True,
                    redo_bad_results=True
                )
    # run(
    #             key=jax.random.PRNGKey(4),
    #             models=[models[1]],
    #             tuning_options=['nuts'],
    #             mh_options = [True],
    #             canonical_options = [True],
    #             langevin_options = [False],
    #             integrator_type_options = ['velocity_verlet'],
    #             diagonal_preconditioning_options = [False],
    #             redo=True,
    #             compute_missing=True,
    #             redo_bad_results=True
    #         )

    # raise Exception

    # run(
    #     key=jax.random.PRNGKey(0),
    #     models=[models[0]],
    #     tuning_options=['grid_search'],
    #     mh_options = [False],
    #     canonical_options = [True],
    #     langevin_options = [True],
    #     integrator_type_options = ['velocity_verlet'],
    #     diagonal_preconditioning_options = [False],
    #     redo=True,
    #     compute_missing=True,
    #     # redo_bad_results=True
    # )
    
    # run(
    #     key=jax.random.PRNGKey(0),
    #     models=[models[0]],
    #     tuning_options=['grid_search'],
    #     mh_options = [True],
    #     canonical_options = [True],
    #     langevin_options = [True],
    #     integrator_type_options = ['velocity_verlet'],
    #     diagonal_preconditioning_options = [False],
    #     redo=True,
    #     compute_missing=True,
    #     # redo_bad_results=True
    # )
    
    # run(
    #     key=jax.random.PRNGKey(0),
    #     models=models,
    #     tuning_options=['nuts'],
    #     mh_options = [True],
    #     canonical_options = [True],
    #     langevin_options = [False],
    #     integrator_type_options = ['velocity_verlet'],
    #     diagonal_preconditioning_options = [True],
    #     redo=False,
    #     compute_missing=True,
    #     # redo_bad_results=True
    # )


    
    # for model in models:

    #     for mh, canonical, langevin, integrator_type in [
    #         (True, True, True, 'velocity_verlet'), 
    #         (True, False, True, 'velocity_verlet'), 
    #         (False, True, True, 'velocity_verlet'), 
    #         (False, False, True, 'mclachlan'),
    #         (True, True, False, 'velocity_verlet'),
            
    #         ]:

    #                         run(
    #                             key=jax.random.PRNGKey(5),
    #                             models=[model],
    #                             tuning_options=['alba'],
    #                             mh_options = [mh],
    #                             canonical_options = [canonical],
    #                             langevin_options = [langevin],
    #                             integrator_type_options = [integrator_type],
    #                             diagonal_preconditioning_options = [False],
    #                             redo=False,
    #                             redo_bad_results=True,
    #                             compute_missing=True,
    #                             # redo_bad_results=True
    #                         )
                            

        
            # print(f"\n\ni={i} \n\n")
            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['alba'],
            #     mh_options = [False],
            #     canonical_options = [False],
            #     langevin_options = [False],
            #     integrator_type_options = ['velocity_verlet'],
            #     diagonal_preconditioning_options = [True],
            #     redo=True,
            #     compute_missing=True,
            #     # redo_bad_results=True
            # )

            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['alba'],
            #     mh_options = [False],
            #     canonical_options = [False],
            #     langevin_options = [True],
            #     integrator_type_options = ['mclachlan'],
            #     diagonal_preconditioning_options = [False],
            #     redo=True,
            #     compute_missing=True,
            #     # redo_bad_results=True
            # )
            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['alba'],
            #     mh_options = [True],
            #     canonical_options = [False],
            #     langevin_options = [True],
            #     integrator_type_options = ['velocity_verlet'],
            #     diagonal_preconditioning_options = [True],
            #     redo=True,
            #     compute_missing=True,
            #     # redo_bad_results=True
            # )
            
            
            
            # run(
            #     key=jax.random.PRNGKey(4),
            #     models=[model],
            #     tuning_options=['nuts'],
            #     mh_options = [True],
            #     canonical_options = [True],
            #     langevin_options = [False],
            #     integrator_type_options = ['velocity_verlet'],
            #     diagonal_preconditioning_options = [False],
            #     redo=False,
            #     compute_missing=True,
            #     # redo_bad_results=True
            # )

          
            # Clear cache after each model to prevent memory accumulation
            # print(f"\n=== Completed model: {model.name} ===")
            # print("Clearing JAX cache after model completion...")
            # clear_jax_cache()
            
            