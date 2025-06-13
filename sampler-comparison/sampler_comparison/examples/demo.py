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
from sampler_evaluation.models.rosenbrock import Rosenbrock
from sampler_evaluation.models.stochastic_volatility_mams_paper import stochastic_volatility_mams_paper

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
        # Rosenbrock(18),
        # stochastic_volatility_mams_paper,
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
    full_results = full_results[full_results['precond'] == 'precond:True']

    full_results_avg = full_results[full_results['max'] == False]
    full_results_max = full_results[full_results['max'] == True]

    
    for model in models:
        results_model_max = full_results_max[full_results_max['Model'] == model.name]
        results_model_avg = full_results_avg[full_results_avg['Model'] == model.name]
        cov_results = results_model_max[results_model_max['statistic'] == 'covariance']
        square_results_max = results_model_max[results_model_max['statistic'] == 'square']
        square_results_avg = results_model_avg[results_model_avg['statistic'] == 'square']

        color_map = {'canonical': 'tab:blue', 'microcanonical': 'tab:orange'}
        hatch_map = {'velocity verlet': '/', 'mclachlan': '\\'}
        alpha_map = {'langevin': 1.0, 'nolangevin': 0.5}

        pretty_name = {
            IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log').name: 'Ill-Conditioned Gaussian in 2D, with condition number 1',
            IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log').name: 'Ill-Conditioned Gaussian in 100D, with condition number 1',
            brownian_motion().name: 'Brownian Motion',
            german_credit().name: 'German Credit',
            item_response().name: 'Item Response',
        }

        # --- Create subplots for max and min ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        for ax, plot_label, plot_df in zip(axes, ['avg', 'max'], [square_results_avg, square_results_max]):
            if plot_df.empty:
                ax.axis('off')
                continue

            mh_order = plot_df['mh'].unique()
            canonical_order = plot_df['canonical'].unique()
            integrator_order = plot_df['integrator_type'].unique()
            langevin_order = plot_df['langevin'].unique()

            bar_width = 0.35
            n_hue = len(canonical_order)
            n_hatch = len(integrator_order)

            # 1. Store x positions for each group
            group_labels = ['With MH Adjustment', 'Without MH Adjustment']
            group_xs = {label: [] for label in group_labels}

            for i, mh in enumerate(mh_order):
                group_label = group_labels[i]
                for j, canonical in enumerate(canonical_order):
                    for k, integrator in enumerate(integrator_order):
                        for l, langevin in enumerate(langevin_order):
                            row = plot_df[
                                (plot_df['mh'] == mh) &
                                (plot_df['canonical'] == canonical) &
                                (plot_df['integrator_type'] == integrator) &
                                (plot_df['langevin'] == langevin)
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
                                group_xs[group_label].append(x)

            # 1. Find the bottom of the bars (should be zero)
            y_avg, y_max = ax.get_ylim()
            ax.set_ylim(0, y_max)  # Ensure y-axis starts at zero

            # 2. Draw brackets below bars, facing upwards (∩)
            bracket_offset = 0.05 * (y_max - 0)
            for label, xs in group_xs.items():
                left = min(xs) - bar_width / (2 * n_hatch)
                right = max(xs) + bar_width / (2 * n_hatch)
                bracket_y = 0 - bracket_offset * 1.5  # below the axis
                bracket_height = 0 - bracket_offset * 0.5
                # Draw upward-facing bracket (∩)
                ax.plot([left, left, (left + right) / 2, right, right],
                        [bracket_y, bracket_height, bracket_height + bracket_offset * 0.5, bracket_height, bracket_y],
                        color='black', lw=1.5)
                # Add label below bracket
                ax.text((left + right) / 2, bracket_y - bracket_offset * 0.2, label, ha='center', va='top', fontsize=12, fontweight='bold')

            # 3. Remove x-tick labels since we have bracket labels
            ax.set_xticklabels([])
            ax.margins(y=0.15)  # Add some space at the bottom for the bracket/label

            ax.set_xlabel('Sampler', fontsize=16, fontweight='bold', labelpad=40)
            if ax is axes[0]:
                ax.set_ylabel('Number of gradients to $b < 0.1$', fontsize=16, fontweight='bold')
            ax.set_title(f"{plot_label.capitalize()}")

        # 4. Place legends only once (on the first axis)
        legend_elements = [
            Patch(facecolor='tab:blue', edgecolor='black', label='canonical'),
            Patch(facecolor='tab:orange', edgecolor='black', label='microcanonical')
        ]
        hatch_elements = [
            Patch(facecolor='white', edgecolor='black', hatch='/', label='Leapfrog'),
            Patch(facecolor='white', edgecolor='black', hatch='\\', label='2nd Order Minimal Norm')
        ]
        alpha_elements = [
            Patch(facecolor='grey', edgecolor='black', alpha=1.0, label='langevin=True'),
            Patch(facecolor='grey', edgecolor='black', alpha=0.5, label='langevin=False')
        ]
        legend1 = axes[1].legend(handles=legend_elements, title='canonical (color)', loc='upper right', bbox_to_anchor=(1, 1))
        legend2 = axes[1].legend(handles=hatch_elements, title='integrator (hatch)', loc='upper left', bbox_to_anchor=(0, 1))
        legend3 = axes[1].legend(handles=alpha_elements, title='langevin (transparency)', loc='upper center', bbox_to_anchor=(0.5, 1))
        axes[1].add_artist(legend1)
        axes[1].add_artist(legend2)
        axes[1].add_artist(legend3)

        plt.suptitle(f"Model: {pretty_name[model.name]}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'sampler_comparison/examples/results_{model.name}_maxmin.png')
        plt.close()


plot_results()