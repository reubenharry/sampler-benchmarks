from itertools import product
import os, sys, inspect
import pandas as pd
import numpy as np
import jax, gc


def base_dir(param_grid):
    folder = 'papers/LAPS/img/trace/'

    for name in param_grid.keys():
        folder += name + '_'
    return folder[:-1] + '/'


def subdir(values):
    file = ''
    for val in values:
        file += str(val) + '_'
    return file[:-1] + '/'


def do_single(func, param_name, param_vals, which, fixed_params=None, verbose=False, extra_word= ''):
    params = {param_name: param_vals[which]}
    # Get the function's default parameter values
    sig = inspect.signature(func)
    default_params = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    # Combine defaults, fixed, and given parameters
    fixed_params = fixed_params or {}
    all_params = {**default_params, **fixed_params, **params}
    # Evaluate
    res = func(**all_params)
    all_params['res'] = res
    base = base_dir(params)
    df = pd.DataFrame([all_params])
    df.to_csv(base + 'data'+extra_word+ str(which)+'.csv', sep= '\t', index= False)


mylogspace = lambda a, b, num, decimals=3: np.round(np.logspace(np.log10(a), np.log10(b), num), decimals)

