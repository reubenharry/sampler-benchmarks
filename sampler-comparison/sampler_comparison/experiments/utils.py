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


model_info = {
    IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 2D, with condition number 1',
        'batch_size' : 128,
        'num_steps' : {True : 5000, False : 5000}},
            
    IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 100D, with condition number 1',
        'batch_size' : 64,
        'num_steps' : {True : 5000, False : 10000}},
    IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 100D, with condition number 1000',
        'batch_size' : 64,
        'num_steps' : {True : 50000, False : 100000}},
    IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log').name: {
        'pretty_name' : 'Ill-Conditioned Gaussian in 10000D, with condition number 100',
        'batch_size' : 16,
        'num_steps' : {True : 50000, False : 100000}},
            
    brownian_motion().name: {
        'pretty_name' : 'Brownian Motion',
        'batch_size' : 128,
        'num_steps' : {True : 40000, False : 100000}},
            
    german_credit().name: {
        'pretty_name' : 'German Credit',
        'batch_size' : 128,
        'num_steps' : {True : 20000, False : 100000}},
    item_response().name: {
        'pretty_name' : 'Item Response',
        'batch_size' : 64,
        'num_steps' : {True : 80000, False : 200000}},
    stochastic_volatility_mams_paper.name: {
        'pretty_name' : 'Stochastic Volatility',
        'batch_size' : 128,
        'num_steps' : {True : 50000, False : 200000}},
    Rosenbrock(18).name: {
        'pretty_name' : 'Rosenbrock',
        'batch_size' : 128,
        'num_steps' : {True : 5000, False : 50000}},
        
    U1(Lt=16, Lx=16, beta=6).name: {
        'pretty_name' : 'U1',
        'batch_size' : 32,
        'num_steps' : {True : 5000, False : 200000}},

    cauchy(ndims=100).name: {
        'pretty_name' : 'Cauchy',
        'batch_size' : 32,
        'num_steps' : {True : 30000, False : 700000}},
    banana().name: {
        'pretty_name' : 'Banana',
        'batch_size' : 128,
        'num_steps' : {True : 20000, False : 50000}}
}