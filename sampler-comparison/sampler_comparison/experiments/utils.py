# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)

sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
sys.path.append("../../blackjax")
# raise Exception("stop")

from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from functools import partial
# import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
from sampler_evaluation.models.gaussian_mams_paper import IllConditionedGaussian
import pandas as pd
from sampler_evaluation.models.banana import banana
from sampler_evaluation.models.brownian import brownian_motion
import os
from results.run_benchmarks import run_benchmarks
# import sampler_evaluation
from sampler_evaluation.models.dirichlet import Dirichlet
import jax
from results.run_benchmarks import lookup_results
import itertools
import jax.numpy as jnp
from sampler_evaluation.models.german_credit import german_credit
from sampler_evaluation.models.item_response import item_response
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper
from sampler_evaluation.models.u1 import U1
from sampler_evaluation.models.cauchy import cauchy
from sampler_evaluation.models.phi4 import phi4
from sampler_evaluation.models.data.estimate_expectations_phi4 import unreduce_lam
from sampler_evaluation.models.bimodal import bimodal_gaussian

model_info = {
    IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 2D, with condition number 1',
        'batch_size' : 1,
        'num_steps' : {True : 5000, False : 5000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 400, False : 400}},
            
    IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 100D, with condition number 1',
        'batch_size' : 64,
        'num_steps' : {True : 5000, False : 10000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 1000, False : 2000}},
    IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 100D, with condition number 1000',
        'batch_size' : 64,
        'num_steps' : {True : 50000, False : 100000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 10000, False : 20000}},
    IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 10000D, with condition number 100',
        'batch_size' : 16,
        'num_steps' : {True : 50000, False : 100000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 10000, False : 20000}},
            

    bimodal_gaussian().name: {
        'pretty_name' : 'Bimodal Gaussian',
        'batch_size' : 128,
        'num_steps' : {True : 50000, False : 100000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 10000, False : 20000}},
    brownian_motion().name: {
        'pretty_name' : 'Brownian Motion',
        'batch_size' : 64,
        'num_steps' : {True : 40000, False : 100000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : True,
        'grid_search_steps' : {True : 30000, False : 30000}},
            
    german_credit().name: {
        'pretty_name' : 'German Credit',
        'batch_size' : 128,
        'num_steps' : {True : 20000, False : 100000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : True,
        'grid_search_steps' : {True : 10000, False : 20000}},
    item_response().name: {
        'pretty_name' : 'Item Response',
        'batch_size' : 64,
        'num_steps' : {True : 80000, False : 200000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : True,
        'grid_search_steps' : {True : 20000, False : 40000}},
    stochastic_volatility_mams_paper.name: {
        'pretty_name' : 'Stochastic Volatility',
        'batch_size' : 128,
        'num_steps' : {True : 50000, False : 200000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : True,
        'grid_search_steps' : {True : 20000, False : 40000}},
    Rosenbrock(18).name: {
        'pretty_name' : 'Rosenbrock',
        'batch_size' : 64,
        'num_steps' : {True : 50000, False : 150000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 40000, False : 40000}},
        
    # U1(Lt=16, Lx=16, beta=6).name: {
    #     'pretty_name' : 'U1',
    #     'batch_size' : 32,
    #     'num_steps' : {True : 5000, False : 200000},
    #     'preferred_statistic' : 'square',
    #     'max_over_parameters' : True,
    #     'grid_search_steps' : {True : 20000, False : 40000}},

    cauchy(ndims=100).name: {
        'pretty_name' : 'Cauchy',
        'batch_size' : 32,
        'num_steps' : {True : 30000, False : 700000},
        'preferred_statistic' : 'entropy',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 70000, False : 140000}},
    banana().name: {
        'pretty_name' : 'Banana',
        'batch_size' : 128,
        'num_steps' : {True : 20000, False : 50000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : True,
        'grid_search_steps' : {True : 5000, False : 10000}},

    phi4(64, unreduce_lam(reduced_lam=4.0, side=64)).name: {
        'pretty_name' : 'Phi4',
        'batch_size' : 8,
        'num_steps' : {True : 500, False : 50000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 40000, False : 40000}},
    
    phi4(128, unreduce_lam(reduced_lam=4.0, side=128)).name: {
        'pretty_name' : 'Phi4',
        'batch_size' : 8,
        'num_steps' : {True : 500, False : 50000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 40000, False : 40000}},
    
    phi4(256, unreduce_lam(reduced_lam=4.0, side=256)).name: {
        'pretty_name' : 'Phi4',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 50000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 40000, False : 40000}},
    
    phi4(512, unreduce_lam(reduced_lam=4.0, side=512)).name: {
        'pretty_name' : 'Phi4',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 50000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 40000, False : 40000}},
    
    phi4(1024, unreduce_lam(reduced_lam=4.0, side=1024)).name: {
        'pretty_name' : 'Phi4',
        'batch_size' : 4,
        'num_steps' : {True : 15000, False : 150000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},

    U1(Lt=64, Lx=64, beta=2.).name: {
        'pretty_name' : 'U1',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 15000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},
    U1(Lt=128, Lx=128, beta=2.).name: {
        'pretty_name' : 'U1',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 15000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},
    U1(Lt=256, Lx=256, beta=2.).name: {
        'pretty_name' : 'U1',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 15000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},
    U1(Lt=512, Lx=512, beta=2.).name: {
        'pretty_name' : 'U1',
        'batch_size' : 4,
        'num_steps' : {True : 500, False : 15000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},
    U1(Lt=1024, Lx=1024, beta=2.).name: {
        'pretty_name' : 'U1',
        'batch_size' : 1,
        'num_steps' : {True : 500, False : 15000},
        'preferred_statistic' : 'square',
        'max_over_parameters' : False,
        'grid_search_steps' : {True : 150000, False : 150000}},
        
        
        
    }

def get_model_specific_preferences(model, is_adjusted, statistic=None, max_over_parameters=None, grid_search_steps=None, num_steps=None):
    """
    Get model-specific preferences for grid search optimization.
    
    Args:
        model: The model to get preferences for
        is_adjusted: Whether this is for an adjusted sampler (True) or unadjusted sampler (False)
        statistic: Which statistic to optimize - if None, will use model-specific preference
        max_over_parameters: Whether to use max_over_parameters - if None, will use model-specific preference
        grid_search_steps: Number of steps for grid search - if None, will use model-specific preference
        num_steps: Total number of steps (used as fallback for grid_search_steps)
    
    Returns:
        Tuple of (statistic, max_over_parameters, grid_search_steps)
    """
    model_name = model.name
    if model_name in model_info:
        model_prefs = model_info[model_name]
        # Use model-specific preferences if not explicitly provided
        if statistic is None:
            statistic = model_prefs.get('preferred_statistic', 'square')
        if max_over_parameters is None:
            max_over_parameters = model_prefs.get('max_over_parameters', True)
        if grid_search_steps is None:
            # Handle both dictionary and scalar grid_search_steps for backward compatibility
            model_grid_search_steps = model_prefs.get('grid_search_steps', None)
            if isinstance(model_grid_search_steps, dict):
                # Use is_adjusted to select the appropriate value
                grid_search_steps = model_grid_search_steps.get(is_adjusted, model_grid_search_steps.get(True, 1000))
            else:
                # Fallback to scalar value or compute from num_steps
                grid_search_steps = model_grid_search_steps if model_grid_search_steps is not None else (num_steps // 10 if num_steps else 1000)
    else:
        # Fallback defaults if model not in model_info
        if statistic is None:
            statistic = 'square'
        if max_over_parameters is None:
            max_over_parameters = True
        if grid_search_steps is None:
            grid_search_steps = num_steps // 10 if num_steps else 1000
    
    return statistic, max_over_parameters, grid_search_steps