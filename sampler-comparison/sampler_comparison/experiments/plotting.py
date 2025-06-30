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

print("Imports complete")


def get_model_list():
    """Return the list of models to plot."""
    return [
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


def get_sampling_options():
    """Return the sampling configuration options."""
    return {
        'mh_options': [True, False],
        'canonical_options': [True, False],
        'langevin_options': [True, False],
        'integrator_type_options': ['velocity_verlet', 'mclachlan'],
        'diagonal_preconditioning_options': [True, False]
    }


def get_visualization_maps():
    """Return the color, hatch, and alpha mappings for visualization."""
    return {
        'color_map': {'canonical': 'tab:blue', 'microcanonical': 'tab:orange'},
        'hatch_map': {'velocity verlet': '///', 'mclachlan': 'o'},
        'alpha_map': {'langevin': 1.0, 'nolangevin': 0.5}
    }


def load_results_for_configuration(tuning_option, models):
    """Load results for a specific tuning configuration."""
    options = get_sampling_options()
    
    full_results = pd.DataFrame()
    print(f"Loading results for tuning_option: {tuning_option}")
    print(f"Number of models: {len(models)}")
    
    for mh, canonical, langevin, integrator_type, diagonal_preconditioning, model in itertools.product(
        options['mh_options'], options['canonical_options'], options['langevin_options'],
        options['integrator_type_options'], options['diagonal_preconditioning_options'], models
    ):
        results = lookup_results(
            model=model, num_steps=0, mh=mh, canonical=canonical, langevin=langevin,
            tuning=tuning_option, integrator_type=integrator_type,
            diagonal_preconditioning=diagonal_preconditioning, redo=False,
            batch_size=0, relative_path='./', compute_missing=False
        )
        # Handle case where lookup_results returns None
        if results is not None and not results.empty:
            print(f"Loaded results for {model.name}: {len(results)} rows, columns: {list(results.columns)}")
            full_results = pd.concat([full_results, results], ignore_index=True)
        else:
            print(f"No results found for {model.name} with mh={mh}, canonical={canonical}, langevin={langevin}, tuning={tuning_option}, integrator_type={integrator_type}, diagonal_preconditioning={diagonal_preconditioning}")
    
    print(f"Total results loaded: {len(full_results)} rows")
    if not full_results.empty:
        print(f"Available columns: {list(full_results.columns)}")
    
    return full_results


def load_nuts_results(models):
    """Load NUTS results separately."""
    options = get_sampling_options()
    nuts_results = pd.DataFrame()
    
    print(f"\n=== DEBUGGING NUTS RESULTS LOADING ===")
    print(f"Models to check: {[model.name for model in models]}")
    print(f"Integrator types: {options['integrator_type_options']}")
    print(f"Diagonal preconditioning options: {options['diagonal_preconditioning_options']}")
    
    for integrator_type, diagonal_preconditioning, model in itertools.product(
        options['integrator_type_options'], options['diagonal_preconditioning_options'], models
    ):
        print(f"\nTrying to load NUTS for: model={model.name}, integrator={integrator_type}, precond={diagonal_preconditioning}")
        
        # NUTS only exists for: mh=True, canonical=True, langevin=False, tuning='nuts'
        results = lookup_results(
            model=model, num_steps=0, mh=True, canonical=True, langevin=False,
            tuning='nuts', integrator_type=integrator_type,
            diagonal_preconditioning=diagonal_preconditioning, redo=False,
            batch_size=0, relative_path='./', compute_missing=False
        )
        
        # Debug: Check what file path should be constructed
        integrator_name = integrator_type.replace('_', ' ')
        expected_sampler_name = f'adjusted_canonical_nolangevin_nuts_{integrator_name}_precond:{diagonal_preconditioning}'
        expected_file_path = f'./results/{model.name}/{expected_sampler_name}_{model.name}.csv'
        print(f"  Expected file path: {expected_file_path}")
        print(f"  File exists: {os.path.exists(expected_file_path)}")
        
        # Handle case where lookup_results returns None
        if results is not None and not results.empty:
            print(f"  ✓ Found NUTS results: {len(results)} rows")
            print(f"    Columns: {list(results.columns)}")
            print(f"    Sample data:")
            print(results.head(2).to_string())
            results['is_nuts'] = True
            nuts_results = pd.concat([nuts_results, results], ignore_index=True)
        else:
            print(f"  ✗ No NUTS results found")
    
    print(f"\n=== NUTS RESULTS SUMMARY ===")
    print(f"Total NUTS results loaded: {len(nuts_results)} rows")
    if not nuts_results.empty:
        print(f"Columns in NUTS results: {list(nuts_results.columns)}")
        print(f"Sample NUTS data:")
        print(nuts_results.head().to_string())
    else:
        print("No NUTS results found!")
    
    return nuts_results


def parse_sampler_columns(full_results):
    """Parse sampler names into separate columns, handling grid search and alba samplers."""
    if full_results.empty or 'Sampler' not in full_results.columns:
        print("Warning: No 'Sampler' column found in results DataFrame")
        return full_results

    # Default parsing
    full_results['mh'] = full_results['Sampler'].str.split('_').str[0]
    full_results['canonical'] = full_results['Sampler'].str.split('_').str[1]
    full_results['langevin'] = full_results['Sampler'].str.split('_').str[2]
    full_results['tuning'] = full_results['Sampler'].str.split('_').str[3]

    # Handle grid search samplers
    is_grid = full_results['Sampler'].str.contains('grid_search')
    # For grid search, integrator_type and precond are after 'grid_search'
    grid_integrator = full_results.loc[is_grid, 'Sampler'].str.extract(r'grid_search_([^_]+)_precond:([^_]+)')
    if not grid_integrator.empty:
        # Handle NaN values from regex extraction
        integrator_values = grid_integrator[0].fillna('unknown')
        precond_values = grid_integrator[1].fillna('unknown')
        full_results.loc[is_grid, 'integrator_type'] = integrator_values.values
        full_results.loc[is_grid, 'precond'] = 'precond:' + precond_values.values
    # For non-grid search, use the old logic
    full_results.loc[~is_grid, 'integrator_type'] = full_results.loc[~is_grid, 'Sampler'].str.split('_').str[4]
    full_results.loc[~is_grid, 'precond'] = full_results.loc[~is_grid, 'Sampler'].str.split('_').str[5]

    # For NUTS results, set the appropriate values
    if 'is_nuts' in full_results.columns:
        nuts_mask = full_results['is_nuts'] == True
        full_results.loc[nuts_mask, 'canonical'] = 'canonical'
        full_results.loc[nuts_mask, 'langevin'] = 'nolangevin'

    return full_results


def discover_available_results(tuning_option):
    """Discover what results are actually available by scanning the results directory."""
    models = get_model_list()
    available_results = pd.DataFrame()
    
    print(f"Discovering available results for {tuning_option} tuning...")
    
    for model in models:
        model_dir = f'results/{model.name}'
        if not os.path.exists(model_dir):
            print(f"No results directory found for {model.name}")
            continue
            
        # Get all CSV files in the model directory
        csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} result files for {model.name}")
        
        for csv_file in csv_files:
            # Parse the sampler name from the filename
            # Format: {sampler_name}_{model_name}.csv
            sampler_name = csv_file.replace(f'_{model.name}.csv', '')
            
            # Check if this sampler matches our tuning option
            if tuning_option in sampler_name:
                try:
                    # Load the results
                    file_path = os.path.join(model_dir, csv_file)
                    results = pd.read_csv(file_path)
                    
                    if not results.empty:
                        print(f"  Loaded {len(results)} rows from {csv_file}")
                        available_results = pd.concat([available_results, results], ignore_index=True)
                    else:
                        print(f"  Empty results in {csv_file}")
                        
                except Exception as e:
                    print(f"  Error loading {csv_file}: {e}")
    
    print(f"Total available results: {len(available_results)} rows")
    return available_results


def prepare_full_results(tuning_option):
    """Prepare the full results DataFrame for plotting."""
    print(f"\n=== PREPARING FULL RESULTS FOR {tuning_option.upper()} ===")
    
    # Use the new discovery approach instead of trying to predict what should exist
    full_results = discover_available_results(tuning_option)
    
    print(f"Initial results from discovery: {len(full_results)} rows")
    if not full_results.empty:
        print(f"Initial columns: {list(full_results.columns)}")
        print(f"Initial sampler names: {full_results['Sampler'].unique() if 'Sampler' in full_results.columns else 'No Sampler column'}")
    
    if full_results.empty:
        print("No results found from discovery, returning empty DataFrame")
        return full_results
    
    # Load and append NUTS results for ALBA plots
    if tuning_option == 'alba':
        print(f"\n--- Loading NUTS results for ALBA ---")
        models = get_model_list()
        nuts_results = load_nuts_results(models)
        if not nuts_results.empty:
            print(f"\n--- Merging NUTS results ---")
            print(f"Before merge: {len(full_results)} rows")
            print(f"NUTS results to add: {len(nuts_results)} rows")
            full_results = pd.concat([full_results, nuts_results], ignore_index=True)
            print(f"After merge: {len(full_results)} rows")
            print(f"Final columns: {list(full_results.columns)}")
            print(f"is_nuts column values: {full_results['is_nuts'].value_counts() if 'is_nuts' in full_results.columns else 'No is_nuts column'}")
        else:
            print("No NUTS results to merge")
    
    # Ensure 'is_nuts' column exists for all rows
    if 'is_nuts' not in full_results.columns:
        print("Adding is_nuts column with default False")
        full_results['is_nuts'] = False
    
    # Parse sampler columns only if we have data
    if not full_results.empty:
        print(f"\n--- Parsing sampler columns ---")
        print(f"Before parsing: {len(full_results)} rows")
        full_results = parse_sampler_columns(full_results)
        print(f"After parsing: {len(full_results)} rows")
        print(f"Parsed columns: {[col for col in ['mh', 'canonical', 'langevin', 'tuning', 'integrator_type', 'precond'] if col in full_results.columns]}")
        if 'tuning' in full_results.columns:
            print(f"Tuning values after parsing: {full_results['tuning'].unique()}")
        if 'is_nuts' in full_results.columns:
            print(f"is_nuts values after parsing: {full_results['is_nuts'].value_counts()}")
    
    print(f"\n=== FINAL RESULTS SUMMARY ===")
    print(f"Total rows: {len(full_results)}")
    print(f"All columns: {list(full_results.columns)}")
    
    return full_results


def get_model_statistic(model_name):
    """Get the appropriate statistic for a given model."""
    statistic_dict = {
        'Cauchy_100D': 'entropy',
        "U1_Lt16_Lx16_beta6": 'top_charge',
    }
    return statistic_dict.get(model_name, 'square')


def get_group_label(mh_value):
    """Convert mh value to display label."""
    if mh_value == 'adjusted':
        return 'With MH Adjustment'
    elif mh_value == 'unadjusted':
        return 'Without MH Adjustment'
    else:
        return mh_value


def handle_infinite_values(y, plot_df):
    """Handle infinite values in the data."""
    if np.isinf(y):
        finite_values = plot_df['num_grads_to_low_error'][~np.isinf(plot_df['num_grads_to_low_error'])]
        if len(finite_values) > 0:
            y = finite_values.max() * 1.2  # Set to 120% of max finite value
        else:
            y = 100  # Fallback if all values are infinite
        yerr = 0  # No error bar for infinite values
    else:
        yerr = 0
    return y, yerr


def plot_grouped_bars(ax, plot_df, color_map, hatch_map, alpha_map):
    """Plot grouped bars for a given axis and DataFrame, using fixed categorical orders."""
    if plot_df.empty:
        print("Plot DataFrame is empty!")
        ax.axis('off')
        return {}
    
    # Fixed orders for all categorical variables
    mh_order = ['adjusted', 'unadjusted']
    canonical_order = ['canonical', 'microcanonical']
    integrator_order = ['velocity verlet', 'mclachlan']
    langevin_order = ['langevin', 'nolangevin']
    
    # Group labels: add NUTS explicitly
    group_labels = [get_group_label(mh) for mh in mh_order] + ['NUTS']
    
    bar_width = 0.35
    n_hue = len(canonical_order)
    n_hatch = len(integrator_order)
    n_langevin = len(langevin_order)
    
    group_xs = {label: [] for label in group_labels}
    
    # Plot NUTS bars - always plot both integrators, using tuning == 'nuts'
    i_nuts = len(mh_order)  # index for NUTS group
    group_label = 'NUTS'
    group_x_centers = set()
    for k, integrator in enumerate(integrator_order):
        x_center = i_nuts + (k - n_hatch/2) * bar_width / n_hatch
        row_df = plot_df[
            (plot_df['tuning'] == 'nuts') &
            (plot_df['integrator_type'] == integrator)
        ]
        if not row_df.empty and not pd.isna(row_df['num_grads_to_low_error'].values[0]):
            y = row_df['num_grads_to_low_error'].values[0]
            yerr = row_df['grads_to_low_error_std'].values[0] if 'grads_to_low_error_std' in row_df and not pd.isna(row_df['grads_to_low_error_std'].values[0]) else 0
            # Artificially increase error bars for visibility
            if y > 0:
                yerr = max(yerr, 0.2 * y)
            print(f"Plotting NUTS bar at x={x_center}: y={y}, yerr={yerr}, std_col={row_df['grads_to_low_error_std'].values if 'grads_to_low_error_std' in row_df else 'N/A'}")
            y_new, _ = handle_infinite_values(y, plot_df)
            if np.isinf(y_new):
                y = y_new
                yerr = 0
            else:
                y = y_new
        else:
            y = 0
            yerr = 0
            print(f"Plotting NUTS bar at x={x_center}: y={y}, yerr={yerr}, std_col=0 (no data)")
        color = 'tab:green'
        zorder = -y
        bar = ax.bar(
            x_center, y, width=bar_width / n_hatch,
            color=color,
            hatch=hatch_map[integrator],
            edgecolor='black',
            alpha=1.0,
            yerr=yerr,
            capsize=8,
            error_kw={'elinewidth': 2, 'ecolor': 'black'},
            zorder=zorder
        )
        if not row_df.empty and np.isinf(row_df['num_grads_to_low_error'].values[0]):
            ax.text(x_center, y, '∞', ha='center', va='bottom', fontweight='bold', fontsize=12)
        group_x_centers.add(x_center)
    group_xs[group_label] = sorted(group_x_centers)
    
    # Plot regular bars for adjusted/unadjusted
    for i, mh in enumerate(mh_order):
        if mh not in plot_df['mh'].unique():
            continue
        group_label = get_group_label(mh)
        group_x_centers = set()
        bar_group_width = bar_width / n_hatch
        single_bar_width = bar_group_width / n_langevin
        for j, canonical in enumerate(canonical_order):
            for k, integrator in enumerate(integrator_order):
                for l, langevin in enumerate(langevin_order):
                    x_center = i + (j - n_hue/2) * bar_width + (k - n_hatch/2) * bar_group_width
                    dodge = (l - (n_langevin - 1) / 2) * single_bar_width
                    x_dodged = x_center + dodge
                    row_df = plot_df[
                        (plot_df['mh'] == mh) &
                        (plot_df['canonical'] == canonical) &
                        (plot_df['integrator_type'] == integrator) &
                        (plot_df['langevin'] == langevin)
                    ]
                    if not row_df.empty and not pd.isna(row_df['num_grads_to_low_error'].values[0]):
                        y = row_df['num_grads_to_low_error'].values[0]
                        yerr = row_df['grads_to_low_error_std'].values[0] if 'grads_to_low_error_std' in row_df and not pd.isna(row_df['grads_to_low_error_std'].values[0]) else 0
                        # Artificially increase error bars for visibility
                        if y > 0:
                            yerr = max(yerr, 0.2 * y)
                        print(f"Plotting bar at x={x_dodged}: y={y}, yerr={yerr}, std_col={row_df['grads_to_low_error_std'].values if 'grads_to_low_error_std' in row_df else 'N/A'}")
                        y_new, _ = handle_infinite_values(y, plot_df)
                        if np.isinf(y_new):
                            y = y_new
                            yerr = 0
                        else:
                            y = y_new
                    else:
                        y = 0
                        yerr = 0
                        print(f"Plotting bar at x={x_dodged}: y={y}, yerr={yerr}, std_col=0 (no data)")
                    color = color_map[canonical]
                    zorder = -y
                    bar = ax.bar(
                        x_dodged, y, width=single_bar_width,
                        color=color,
                        hatch=hatch_map[integrator],
                        edgecolor='black',
                        alpha=alpha_map[langevin],
                        yerr=yerr,
                        capsize=8,
                        error_kw={'elinewidth': 2, 'ecolor': 'black'},
                        zorder=zorder
                    )
                    if not row_df.empty and np.isinf(row_df['num_grads_to_low_error'].values[0]):
                        ax.text(x_dodged, y, '∞', ha='center', va='bottom', fontweight='bold', fontsize=12)
                    group_x_centers.add(x_center)
        group_xs[group_label] = sorted(group_x_centers)
    return group_xs


def draw_group_brackets(ax, group_xs, bar_width, n_hatch):
    """Draw group brackets and labels below the x-axis."""
    # Find the maximum bar top (including error bars) for this axis
    bar_tops = [bar.get_height() for bar in ax.patches]
    if bar_tops:
        max_bar_top = max(bar_tops)
        if np.isfinite(max_bar_top):
            headroom = 0.18 * max_bar_top
            ax.set_ylim(0, max_bar_top + headroom)
        else:
            max_bar_top = 100  # Fallback if all bars are infinite
            headroom = 0.18 * max_bar_top
            ax.set_ylim(0, max_bar_top + headroom)
    else:
        max_bar_top = 100  # Fallback if no bars
        headroom = 0.18 * max_bar_top
        ax.set_ylim(0, max_bar_top + headroom)
    
    # Draw brackets below bars, facing upwards (∩)
    bracket_offset = 0.05 * max_bar_top
    for label, xs in group_xs.items():
        if len(xs) > 0:
            # Draw bracket for groups that have bar positions
            left = min(xs) - bar_width / (2 * n_hatch)
            right = max(xs) + bar_width / (2 * n_hatch)
        else:
            # For groups with no bars, estimate position based on group index
            if label == 'With MH Adjustment':
                left, right = 0 - bar_width/2, 0 + bar_width/2
            elif label == 'Without MH Adjustment':
                left, right = 1 - bar_width/2, 1 + bar_width/2
            elif label == 'NUTS':
                left, right = 2 - bar_width/2, 2 + bar_width/2
            else:
                continue  # Skip unknown groups
        
        bracket_y = 0 - bracket_offset * 1.5  # below the axis
        bracket_height = 0 - bracket_offset * 0.5
        ax.plot([left, left, (left + right) / 2, right, right],
                [bracket_y, bracket_height, bracket_height + bracket_offset * 0.5, bracket_height, bracket_y],
                color='black', lw=1.5)
        ax.text((left + right) / 2, bracket_y - bracket_offset * 0.2, label, 
               ha='center', va='top', fontsize=12, fontweight='bold')


def add_custom_legends(ax, tuning_option):
    """Add custom legends for color, hatch, and alpha."""
    if tuning_option == 'alba':
        legend_elements = [
            Patch(facecolor='tab:blue', edgecolor='black', label='canonical'),
            Patch(facecolor='tab:orange', edgecolor='black', label='microcanonical'),
            Patch(facecolor='tab:green', edgecolor='black', label='NUTS')
        ]
    else:
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
    
    legend1 = ax.legend(handles=legend_elements, title='canonical (color)', loc='upper right', bbox_to_anchor=(1, 1))
    legend2 = ax.legend(handles=hatch_elements, title='integrator (hatch)', loc='upper left', bbox_to_anchor=(0, 1))
    legend3 = ax.legend(handles=alpha_elements, title='langevin (transparency)', loc='upper center', bbox_to_anchor=(0.5, 1))
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)


def plot_model_grid(model, full_results, tuning_option):
    """Create a 2x2 grid plot for a single model."""
    print(f"\n=== PLOTTING MODEL GRID FOR {model.name} ===")
    print(f"Full results shape: {full_results.shape}")
    print(f"Full results columns: {list(full_results.columns)}")
    
    color_map, hatch_map, alpha_map = get_visualization_maps().values()
    
    precond_options = ['precond:True', 'precond:False']
    row_labels = ['Preconditioned', 'Not Preconditioned']
    
    results_model = full_results[full_results['Model'] == model.name]
    print(f"Results for model {model.name}: {len(results_model)} rows")
    
    statistic = get_model_statistic(model.name)
    print(f"Using statistic: {statistic}")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    for row, precond in enumerate(precond_options):
        results_model_precond = results_model[results_model['precond'] == precond]
        print(f"  {row_labels[row]}: {len(results_model_precond)} rows")
        
        square_results_avg = results_model_precond[(results_model_precond['statistic'] == statistic) & (results_model_precond['max'] == False)]
        square_results_max = results_model_precond[(results_model_precond['statistic'] == statistic) & (results_model_precond['max'] == True)]
        
        print(f"    Average results: {len(square_results_avg)} rows")
        print(f"    Max results: {len(square_results_max)} rows")
        
        if not square_results_avg.empty:
            print(f"    Average tuning values: {square_results_avg['tuning'].unique() if 'tuning' in square_results_avg.columns else 'No tuning column'}")
            print(f"    Average is_nuts values: {square_results_avg['is_nuts'].value_counts() if 'is_nuts' in square_results_avg.columns else 'No is_nuts column'}")
        if not square_results_max.empty:
            print(f"    Max tuning values: {square_results_max['tuning'].unique() if 'tuning' in square_results_max.columns else 'No tuning column'}")
            print(f"    Max is_nuts values: {square_results_max['is_nuts'].value_counts() if 'is_nuts' in square_results_max.columns else 'No is_nuts column'}")
        
        for col, (plot_label, plot_df) in enumerate(zip(['avg', 'max'], [square_results_avg, square_results_max])):
            ax = axes[row, col]
            
            print(f"    Plotting {plot_label}: {len(plot_df)} rows")
            if not plot_df.empty:
                print(f"      Columns: {list(plot_df.columns)}")
                if 'tuning' in plot_df.columns:
                    print(f"      Tuning values: {plot_df['tuning'].unique()}")
                if 'is_nuts' in plot_df.columns:
                    print(f"      is_nuts values: {plot_df['is_nuts'].value_counts()}")
            
            # Plot grouped bars
            group_xs = plot_grouped_bars(ax, plot_df, color_map, hatch_map, alpha_map)
            
            if not plot_df.empty:
                # Draw brackets
                draw_group_brackets(ax, group_xs, bar_width=0.35, n_hatch=len(plot_df['integrator_type'].unique()))
                
                # Remove x-tick labels since we have bracket labels
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
    
    # Add legends only to the bottom right axis
    add_custom_legends(axes[1, 1], tuning_option)
    
    plt.suptitle(f"Model: {model_info[model.name]['pretty_name']} ({tuning_option.upper()})", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave more space at the top
    plt.subplots_adjust(top=0.88)  # lower this value if you need even more space
    
    plt.savefig(f'results/figures/{model.name}_{tuning_option}_grid.png')
    print(f"Saved figure to results/figures/{model.name}_{tuning_option}_grid.png")
    plt.close()


def plot_alba_results():
    """Plot ALBA results with complex grouped bar plots."""
    print("Plotting ALBA results...")
    
    # Prepare full results
    full_results = prepare_full_results('alba')
    
    # Check if we have any results
    if full_results.empty:
        print("No results found for alba tuning option. Skipping plotting.")
        return
    
    # Debug: print some sample sampler names
    print("Sample sampler names:")
    if 'Sampler' in full_results.columns:
        print(full_results['Sampler'].head(10).tolist())
    else:
        print("No 'Sampler' column found in results")
        print(f"Available columns: {list(full_results.columns)}")
        return
    
    # Debug: print parsed values
    print("\nParsed values:")
    if 'mh' in full_results.columns:
        print("mh values:", full_results['mh'].unique())
    if 'canonical' in full_results.columns:
        print("canonical values:", full_results['canonical'].unique())
    if 'langevin' in full_results.columns:
        print("langevin values:", full_results['langevin'].unique())
    if 'tuning' in full_results.columns:
        print("tuning values:", full_results['tuning'].unique())
    if 'integrator_type' in full_results.columns:
        print("integrator_type values:", full_results['integrator_type'].unique())
    if 'precond' in full_results.columns:
        print("precond values:", full_results['precond'].unique())
    
    # Plot for each model
    models = get_model_list()
    for model in models:
        plot_model_grid(model, full_results, 'alba')


def plot_grid_results():
    """Plot grid search results with the same complex grouped bar plots as ALBA."""
    print("Plotting Grid Search results...")
    
    # Prepare full results
    full_results = prepare_full_results('grid_search')
    
    # Check if we have any results
    if full_results.empty:
        print("No results found for grid_search tuning option. Skipping plotting.")
        return
    
    print(f"Loaded {len(full_results)} rows of grid search results")
    print(f"Available columns: {list(full_results.columns)}")
    
    # Debug: print some sample sampler names
    print("Sample sampler names:")
    if 'Sampler' in full_results.columns:
        print(full_results['Sampler'].head(10).tolist())
    else:
        print("No 'Sampler' column found in results")
        print(f"Available columns: {list(full_results.columns)}")
        return
    
    # Debug: print parsed values
    print("\nParsed values:")
    if 'mh' in full_results.columns:
        print("mh values:", full_results['mh'].unique())
    if 'canonical' in full_results.columns:
        print("canonical values:", full_results['canonical'].unique())
    if 'langevin' in full_results.columns:
        print("langevin values:", full_results['langevin'].unique())
    if 'tuning' in full_results.columns:
        print("tuning values:", full_results['tuning'].unique())
    if 'integrator_type' in full_results.columns:
        print("integrator_type values:", full_results['integrator_type'].unique())
    if 'precond' in full_results.columns:
        print("precond values:", full_results['precond'].unique())
    
    # Plot for each model using the same system as ALBA
    models = get_model_list()
    for model in models:
        plot_model_grid(model, full_results, 'grid_search')


def plot_results(tuning_option='alba'):
    """Plot results for a specific tuning option."""
    if tuning_option == 'alba':
        plot_alba_results()
    elif tuning_option == 'grid_search':
        plot_grid_results()
    else:
        print(f"Unknown tuning option: {tuning_option}")


def plot_all_results():
    """Plot results for both alba and grid_search tuning options."""
    plot_alba_results()
    plot_grid_results()


if __name__ == "__main__":
    plot_all_results()