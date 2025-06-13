# from sampler_comparison.samplers.hamiltonianmontecarlo.nuts import nuts
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)

batch_size = 512
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)
num_cores = jax.local_device_count()

# print(os.listdir("../../../sampler-comparison"))

# import sys
sys.path.append("../sampler-comparison")
sys.path.append("../sampler-evaluation")
sys.path.append("../../src/inference-gym/spinoffs/inference_gym")
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_results():

    mh_options = [True, False]
    canonical_options = [True, False]
    langevin_options = [True, False]
    tuning_options = ['alba']
    integrator_type_options = ['velocity_verlet', 'mclachlan']
    diagonal_preconditioning_options = [True, False]
    models = [
        IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log'),
        brownian_motion(),
        german_credit(),
        item_response(),
        ]

    redo = False 

    full_results = pd.DataFrame()
    for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
        results = lookup_results(model=model, num_steps=20000, mh=mh, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=redo, batch_size=512, relative_path='./')
        full_results = pd.concat([full_results, results], ignore_index=True)

    full_results['mh'] = full_results['Sampler'].str.split('_').str[0]
    full_results['canonical'] = full_results['Sampler'].str.split('_').str[1]
    full_results['langevin'] = full_results['Sampler'].str.split('_').str[2]
    full_results['tuning'] = full_results['Sampler'].str.split('_').str[3]
    full_results['integrator_type'] = full_results['Sampler'].str.split('_').str[4]
    full_results['precond'] = full_results['Sampler'].str.split('_').str[5]
    full_results = full_results[full_results['max'] == False]
    full_results = full_results[full_results['precond'] == 'precond:True']

    
    for model in models:
        results_model = full_results[full_results['Model'] == model.name]
        cov_results = results_model[results_model['statistic'] == 'covariance']
        square_results = results_model[results_model['statistic'] == 'square']

        color_map = {'canonical': 'tab:blue', 'microcanonical': 'tab:orange'}
        hatch_map = {'velocity verlet': '/', 'mclachlan': '\\'}
        alpha_map = {'langevin': 1.0, 'nolangevin': 0.5}

        df = square_results

        mh_order = df['mh'].unique()
        canonical_order = df['canonical'].unique()
        integrator_order = df['integrator_type'].unique()
        langevin_order = df['langevin'].unique()

        bar_width = 0.35
        n_hue = len(canonical_order)
        n_hatch = len(integrator_order)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bars with error bars
        for i, mh in enumerate(mh_order):
            for j, canonical in enumerate(canonical_order):
                for k, integrator in enumerate(integrator_order):
                    for l, langevin in enumerate(langevin_order):
                        row = df[
                            (df['mh'] == mh) &
                            (df['canonical'] == canonical) &
                            (df['integrator_type'] == integrator) &
                            (df['langevin'] == langevin)
                        ]
                        if not row.empty:
                            y = row['num_grads_to_low_error'].values[0]
                            yerr = row['grads_to_low_error_std'].values[0] if 'grads_to_low_error_std' in row else 0
                            x = i + (j - n_hue/2) * bar_width + (k - n_hatch/2) * (bar_width / n_hatch)
                            ax.bar(
                                x, y, width=bar_width / n_hatch,
                                color=color_map[canonical],
                                hatch=hatch_map[integrator],
                                edgecolor='black',
                                alpha=alpha_map[langevin],
                                yerr=yerr,
                                capsize=4
                            )

        ax.set_xticks(np.arange(len(mh_order)))
        ax.set_xticklabels(mh_order)
        ax.set_xlabel('Adjustment')
        ax.set_ylabel('Number of Gradients to $b < 0.01$')
        ax.set_title(f'Model: {model.name}')

        # Build simplified legends
        legend_elements = [
            Patch(facecolor='tab:blue', edgecolor='black', label='canonical'),
            Patch(facecolor='tab:orange', edgecolor='black', label='microcanonical')
        ]
        hatch_elements = [
            Patch(facecolor='white', edgecolor='black', hatch='/', label='velocity_verlet'),
            Patch(facecolor='white', edgecolor='black', hatch='\\', label='mclachlan')
        ]
        alpha_elements = [
            Patch(facecolor='grey', edgecolor='black', alpha=1.0, label='langevin=True'),
            Patch(facecolor='grey', edgecolor='black', alpha=0.5, label='langevin=False')
        ]

        # Place legends
        legend1 = ax.legend(handles=legend_elements, title='canonical (color)', loc='upper right')
        legend2 = ax.legend(handles=hatch_elements, title='integrator (hatch)', loc='upper center')
        legend3 = ax.legend(handles=alpha_elements, title='langevin (transparency)', loc='upper left')
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        ax.add_artist(legend3)

        plt.tight_layout()
        plt.savefig(f'sampler_comparison/examples/results_{model.name}.png')
        plt.close()


plot_results()