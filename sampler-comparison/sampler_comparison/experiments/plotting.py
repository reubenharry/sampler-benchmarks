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
sys.path.append("../../blackjax")
# raise Exception("stop")

from sampler_comparison.samplers.microcanonicalmontecarlo.adjusted import (
    adjusted_mclmc,
)
from sampler_comparison.experiments.utils import model_info
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
from sampler_evaluation.models.cauchy import cauchy
from sampler_evaluation.models.u1 import U1


def plot_results():

    mh_options = [True, False]
    canonical_options = [True, False]
    langevin_options = [True, False]
    tuning_options = ['alba']
    integrator_type_options = ['velocity_verlet', 'mclachlan']
    # integrator_type_options = ['velocity_verlet']
    diagonal_preconditioning_options = [True, False]

    # keys of model_info
    models = [
        IllConditionedGaussian(ndims=2, condition_number=1, eigenvalues='log'),
        IllConditionedGaussian(ndims=100, condition_number=1000, eigenvalues='log', do_covariance=False),
        IllConditionedGaussian(ndims=100, condition_number=1, eigenvalues='log', do_covariance=False),
        IllConditionedGaussian(ndims=10000, condition_number=100, eigenvalues='log', do_covariance=False),
        brownian_motion(),
        german_credit(),
        item_response(),
        Rosenbrock(18),
        stochastic_volatility_mams_paper,
        cauchy(ndims=100),
        U1(Lt=16, Lx=16, beta=6),
        banana(),
        ]

   
    full_results = pd.DataFrame()
    for mh, canonical, langevin, tuning, integrator_type, diagonal_preconditioning, model in itertools.product(mh_options, canonical_options, langevin_options, tuning_options, integrator_type_options, diagonal_preconditioning_options, models):
        results = lookup_results(model=model, num_steps=0, mh=mh, canonical=canonical, langevin=langevin, tuning=tuning, integrator_type=integrator_type, diagonal_preconditioning=diagonal_preconditioning, redo=False, batch_size=0, relative_path='./', compute_missing=False)
        full_results = pd.concat([full_results, results], ignore_index=True)

    full_results['mh'] = full_results['Sampler'].str.split('_').str[0]
    full_results['canonical'] = full_results['Sampler'].str.split('_').str[1]
    full_results['langevin'] = full_results['Sampler'].str.split('_').str[2]
    full_results['tuning'] = full_results['Sampler'].str.split('_').str[3]
    full_results['integrator_type'] = full_results['Sampler'].str.split('_').str[4]
    full_results['precond'] = full_results['Sampler'].str.split('_').str[5]

    full_results_avg = full_results[full_results['max'] == False]
    full_results_max = full_results[full_results['max'] == True]

    
    precond_options = ['precond:True', 'precond:False']
    row_labels = ['Preconditioned', 'Not Preconditioned']

    for model in models:
        results_model_max = full_results_max[full_results_max['Model'] == model.name]
        results_model_avg = full_results_avg[full_results_avg['Model'] == model.name]
        cov_results = results_model_max[results_model_max['statistic'] == 'covariance']
        square_results_max = results_model_max[results_model_max['statistic'] == 'square']
        square_results_avg = results_model_avg[results_model_avg['statistic'] == 'square']

        color_map = {'canonical': 'tab:blue', 'microcanonical': 'tab:orange'}
        hatch_map = {'velocity verlet': '///', 'mclachlan': 'o'}
        alpha_map = {'langevin': 1.0, 'nolangevin': 0.5}

        

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        for row, precond in enumerate(precond_options):
            results_model = full_results[full_results['Model'] == model.name]
            results_model = results_model[results_model['precond'] == precond]
            
            # Choose appropriate statistic based on model
            statistic_dict = {
                'Cauchy_100D' : 'entropy',
                "U1_Lt16_Lx16_beta6" : 'top_charge',
            }
            statistic = statistic_dict.get(model.name, 'square')
            # 'entropy' if 'Cauchy' in model.name else 'square'
            
            square_results_avg = results_model[(results_model['statistic'] == statistic) & (results_model['max'] == False)]
            square_results_max = results_model[(results_model['statistic'] == statistic) & (results_model['max'] == True)]
            
            # print(f"Square results avg rows: {len(square_results_avg)}")
            # print(f"Square results max rows: {len(square_results_max)}")
            # print("\nSample of values:")
            # print("Average values:", square_results_avg['num_grads_to_low_error'].head())
            # print("Max values:", square_results_max['num_grads_to_low_error'].head())
            
            for col, (plot_label, plot_df) in enumerate(zip(['avg', 'max'], [square_results_avg, square_results_max])):
                ax = axes[row, col]
                # print(f"\nPlotting {plot_label} data:")
                # print(f"Number of rows in plot_df: {len(plot_df)}")
                if plot_df.empty:
                    # print("Plot DataFrame is empty!")
                    ax.axis('off')
                    continue

                # Create a DataFrame with all possible combinations
                all_combos = pd.DataFrame(
                    list(itertools.product(
                        ["adjusted", "unadjusted"],
                        ["canonical", "microcanonical"],
                        ["velocity verlet", "mclachlan"],
                        ["langevin", "nolangevin"]
                    )),
                    columns=["mh", "canonical", "integrator_type", "langevin"]
                )
                
                # Merge with plot_df to ensure all combinations are present
                plot_df_full = pd.merge(
                    all_combos,
                    plot_df,
                    on=["mh", "canonical", "integrator_type", "langevin"],
                    how="left"
                )
                # print("\nAfter merging with all combinations:")
                # print(f"Number of rows in plot_df_full: {len(plot_df_full)}")
                # print("Sample of merged data:")
                # print(plot_df_full[['mh', 'canonical', 'integrator_type', 'langevin', 'num_grads_to_low_error']].head())

                # Always use the full set of possible values for all variables
                mh_order = ["adjusted", "unadjusted"]
                group_labels = ['With MH Adjustment', 'Without MH Adjustment']
                canonical_order = ['canonical', 'microcanonical']
                integrator_order = ['velocity verlet', 'mclachlan']
                langevin_order = ['langevin', 'nolangevin']

                bar_width = 0.35
                n_hue = len(canonical_order)
                n_hatch = len(integrator_order)
                n_langevin = len(langevin_order)
                bar_group_width = bar_width / n_hatch
                single_bar_width = bar_group_width / n_langevin

                group_xs = {label: [] for label in group_labels}

                for i, mh in enumerate(mh_order):
                    group_label = group_labels[i]
                    group_x_centers = set()
                    for j, canonical in enumerate(canonical_order):
                        for k, integrator in enumerate(integrator_order):
                            for l, langevin in enumerate(langevin_order):
                                x_center = i + (j - n_hue/2) * bar_width + (k - n_hatch/2) * bar_group_width
                                dodge = (l - (n_langevin - 1) / 2) * single_bar_width
                                x_dodged = x_center + dodge
                                row_df = plot_df_full[
                                    (plot_df_full['mh'] == mh) &
                                    (plot_df_full['canonical'] == canonical) &
                                    (plot_df_full['integrator_type'] == integrator) &
                                    (plot_df_full['langevin'] == langevin)
                                ]
                                if not row_df.empty and not pd.isna(row_df['num_grads_to_low_error'].values[0]):
                                    y = row_df['num_grads_to_low_error'].values[0]
                                    # print(f"\nPlotting bar with y={y} (before inf handling)")
                                    yerr = row_df['grads_to_low_error_std'].values[0] if 'grads_to_low_error_std' in row_df and not pd.isna(row_df['grads_to_low_error_std'].values[0]) else 0
                                    
                                    # Handle infinite values
                                    if np.isinf(y):
                                        # print("Found infinite value!")
                                        # Get the maximum finite value in the dataset for scaling
                                        finite_values = plot_df_full['num_grads_to_low_error'][~np.isinf(plot_df_full['num_grads_to_low_error'])]
                                        # print(f"Number of finite values found: {len(finite_values)}")
                                        if len(finite_values) > 0:
                                            y = finite_values.max() * 1.2  # Set to 120% of max finite value
                                            # print(f"Setting to 120% of max finite value: {y}")
                                        else:
                                            y = 100  # Fallback if all values are infinite
                                            # print("All values infinite, using fallback value: 100")
                                        yerr = 0  # No error bar for infinite values
                                else:
                                    y = 0
                                    yerr = 0
                                color = color_map[canonical]
                                zorder = -y
                                bar = ax.bar(
                                    x_dodged, y, width=single_bar_width,
                                    color=color,
                                    hatch=hatch_map[integrator],
                                    edgecolor='black',
                                    alpha=alpha_map[langevin],
                                    yerr=yerr,
                                    capsize=4,
                                    zorder=zorder
                                )
                                
                                # Add infinity symbol for infinite values
                                if not row_df.empty and np.isinf(row_df['num_grads_to_low_error'].values[0]):
                                    ax.text(x_dodged, y, '∞', 
                                          ha='center', va='bottom',
                                          fontweight='bold', fontsize=12)
                                group_x_centers.add(x_center)  # always add, regardless of data
                    group_xs[group_label] = sorted(group_x_centers)

                # 1. Find the maximum bar top (including error bars) for this axis
                bar_tops = [bar.get_height() for bar in ax.patches]
                if bar_tops and all(len(xs) > 0 for xs in group_xs.values()):
                    max_bar_top = max(bar_tops)
                    if not np.isfinite(max_bar_top):
                        pass  # Do not turn off axis
                    else:
                        headroom = 0.18 * max_bar_top
                        ax.set_ylim(0, max_bar_top + headroom)

                        # 2. Draw brackets below bars, facing upwards (∩)
                        bracket_offset = 0.05 * max_bar_top
                        for label, xs in group_xs.items():
                            # Always draw the bracket, even if all bars are zero (xs will still have positions)
                            left = min(xs) - bar_width / (2 * n_hatch)
                            right = max(xs) + bar_width / (2 * n_hatch)
                            bracket_y = 0 - bracket_offset * 1.5  # below the axis
                            bracket_height = 0 - bracket_offset * 0.5
                            ax.plot([left, left, (left + right) / 2, right, right],
                                    [bracket_y, bracket_height, bracket_height + bracket_offset * 0.5, bracket_height, bracket_y],
                                    color='black', lw=1.5)
                            ax.text((left + right) / 2, bracket_y - bracket_offset * 0.2, label, ha='center', va='top', fontsize=12, fontweight='bold')

                # 3. Remove x-tick labels since we have bracket labels
                ax.set_xticklabels([])
                ax.margins(y=0.15)  # Add some space at the bottom for the bracket/label

                # Always set axis labels and title, even if axis is empty
                ax.set_title(f"{row_labels[row]}, {plot_label.capitalize()}")
                if col == 0:
                    ax.set_ylabel('Number of gradients to $b_{\\mathit{avg}}(x^2) < 0.1$', fontsize=16, fontweight='bold')
                else:
                    ax.set_ylabel('')
                if row == 1:
                    ax.set_xlabel('Sampler', fontsize=16, fontweight='bold', labelpad=40)
                else:
                    ax.set_xlabel('')

        # Place legends only once (on the bottom right axis)
        legend_elements = [
            Patch(facecolor='tab:blue', edgecolor='black', label='canonical'),
            Patch(facecolor='tab:orange', edgecolor='black', label='microcanonical')
        ]
        hatch_elements = [
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Leapfrog', linewidth=2),
            Patch(facecolor='white', edgecolor='black', hatch='o', label='2nd Order Minimal Norm', linewidth=2)
        ]
        alpha_elements = [
            Patch(facecolor='grey', edgecolor='black', alpha=1.0, label='langevin=True'),
            Patch(facecolor='grey', edgecolor='black', alpha=0.5, label='langevin=False')
        ]
        legend1 = axes[1, 1].legend(handles=legend_elements, title='canonical (color)', loc='upper right', bbox_to_anchor=(1, 1))
        legend2 = axes[1, 1].legend(handles=hatch_elements, title='integrator (hatch)', loc='upper left', bbox_to_anchor=(0, 1))
        legend3 = axes[1, 1].legend(handles=alpha_elements, title='langevin (transparency)', loc='upper center', bbox_to_anchor=(0.5, 1))
        axes[1, 1].add_artist(legend1)
        axes[1, 1].add_artist(legend2)
        axes[1, 1].add_artist(legend3)

        plt.suptitle(f"Model: {model_info[model.name]['pretty_name']}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave more space at the top

        plt.subplots_adjust(top=0.88)  # lower this value if you need even more space
        plt.savefig(f'results/figures/{model.name}_precond_grid.png')
        print(f"Saved figure to results/figures/{model.name}_precond_grid.png")
        plt.close()

if __name__ == "__main__":
    plot_results()