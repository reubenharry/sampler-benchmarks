import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy.ma as ma

# Generate synthetic data
# We'll create 16 configurations (2^4) with random values
np.random.seed(42)  # for reproducibility

# Create all possible combinations of 4 binary variables
x_values = np.array([0, 1])
y_values = np.array([0, 1])
z_values = np.array([0, 1])
w_values = np.array([0, 1])  # This will be used for coloring

# Create a grid of all possible combinations
X, Y, Z, W = np.meshgrid(x_values, y_values, z_values, w_values)
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()  # Not used for bar base, just for config
W = W.flatten()

# Generate random values for each configuration
values = np.random.normal(0, 1, len(X))

# Bar settings
bar_width = 0.6
bar_depth = 0.6
bar_base = np.zeros_like(X)  # All bars start at z=0

# Color array: red for W=1, blue for W=0, with alpha
colors = np.array([[1, 0, 0, 0.7] if w == 1 else [0, 0.2, 1, 0.7] for w in W])

# Create the figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

bars = ax.bar3d(X, Y, bar_base, bar_width, bar_depth, values, color=colors, shade=True)

# Optional: Add value labels on top of each bar
for x, y, v in zip(X, Y, values):
    ax.text(x + bar_width/2, y + bar_depth/2, max(0, v) + 0.05, f'{v:.2f}',
            ha='center', va='bottom', fontsize=8, color='black')

# Set labels
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.set_zlabel('Value')
ax.set_title('3D Bar Chart of Binary Variables\n(Red = Variable 4 True, Blue = Variable 4 False)')

# Set axis limits
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(min(0, values.min() - 0.5), values.max() + 0.5)

# Add a colorbar legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=(1, 0, 0, 0.7), label='Variable 4 = True'),
    Patch(facecolor=(0, 0.2, 1, 0.7), label='Variable 4 = False')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()

# Example DataFrame (replace with your actual data)
df = pd.DataFrame({
    'mh': ['A', 'A', 'B', 'B'],
    'canonical': ['yes', 'no', 'yes', 'no'],
    'integrator': ['euler', 'rk4', 'euler', 'rk4'],
    'langevin': ['langevin', 'nolangevin', 'langevin', 'nolangevin'],
    'num_grads_to_low_error': [10, 12, 8, 15]
})

# Define color, hatch, and alpha maps
color_map = {'yes': 'tab:blue', 'no': 'tab:orange'}
hatch_map = {'euler': '/', 'rk4': '\\'}
alpha_map = {'langevin': 1.0, 'nolangevin': 0.5}

mh_order = df['mh'].unique()
canonical_order = df['canonical'].unique()
integrator_order = df['integrator'].unique()
langevin_order = df['langevin'].unique()

bar_width = 0.35
n_hue = len(canonical_order)
n_hatch = len(integrator_order)

fig, ax = plt.subplots(figsize=(10, 6))

for i, mh in enumerate(mh_order):
    for j, canonical in enumerate(canonical_order):
        for k, integrator in enumerate(integrator_order):
            for l, langevin in enumerate(langevin_order):
                row = df[
                    (df['mh'] == mh) &
                    (df['canonical'] == canonical) &
                    (df['integrator'] == integrator) &
                    (df['langevin'] == langevin)
                ]
                if not row.empty:
                    y = row['num_grads_to_low_error'].values[0]
                    x = i + (j - n_hue/2) * bar_width + (k - n_hatch/2) * (bar_width / n_hatch)
                    ax.bar(
                        x, y, width=bar_width / n_hatch,
                        color=color_map[canonical],
                        hatch=hatch_map[integrator],
                        edgecolor='black',
                        alpha=alpha_map[langevin],
                        label=f"{canonical}, {integrator}, {langevin}" if i == 0 else ""
                    )

ax.set_xticks(np.arange(len(mh_order)))
ax.set_xticklabels(mh_order)
ax.set_xlabel('mh')
ax.set_ylabel('num_grads_to_low_error')
ax.set_title('Barplot with Color (canonical), Hatch (integrator), and Alpha (langevin)')

# Build custom legend
from matplotlib.patches import Patch
legend_elements = []
for canonical in canonical_order:
    for integrator in integrator_order:
        for langevin in langevin_order:
            legend_elements.append(
                Patch(
                    facecolor=color_map[canonical],
                    hatch=hatch_map[integrator],
                    edgecolor='black',
                    alpha=alpha_map[langevin],
                    label=f"{canonical}, {integrator}, {langevin}"
                )
            )
ax.legend(handles=legend_elements, title='canonical, integrator, langevin')

plt.tight_layout()
plt.show()

def preprocess_values(values, cap_value=None):
    """
    Preprocess values to handle infinities and NaNs.
    If cap_value is provided, values above it will be capped.
    Returns processed values and a mask indicating which values were infinite.
    """
    inf_mask = np.isinf(values)
    nan_mask = np.isnan(values)
    
    # Create a masked array
    processed_values = ma.array(values, mask=inf_mask | nan_mask)
    
    if cap_value is not None:
        # Cap finite values at cap_value
        finite_mask = ~(inf_mask | nan_mask)
        processed_values[finite_mask] = np.minimum(processed_values[finite_mask], cap_value)
        
        # Set infinite values to cap_value
        processed_values[inf_mask] = cap_value
    
    return processed_values, inf_mask

def plot_results(df, model_name, output_file=None):
    """Plot the results with proper handling of infinite values"""
    # Define visual encoding maps
    color_map = {'canonical': 'tab:blue', 'microcanonical': 'tab:orange'}
    hatch_map = {'Leapfrog': '/', '2nd Order Minimal Norm': '\\'}
    alpha_map = {'langevin': 1.0, 'nolangevin': 0.5}
    
    # Get unique values for each category
    mh_order = sorted(df['mh'].unique())
    canonical_order = sorted(df['canonical'].unique())
    integrator_order = sorted(df['integrator'].unique())
    langevin_order = sorted(df['langevin'].unique())
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar positioning parameters
    bar_width = 0.35
    n_hue = len(canonical_order)
    n_hatch = len(integrator_order)
    
    # Find a reasonable cap value for infinite results
    finite_values = df['value'][~np.isinf(df['value'])]
    if len(finite_values) > 0:
        cap_value = np.percentile(finite_values, 95) * 1.2  # 20% above 95th percentile
    else:
        cap_value = 100  # fallback if all values are infinite
    
    # Plot bars
    for i, mh in enumerate(mh_order):
        for j, canonical in enumerate(canonical_order):
            for k, integrator in enumerate(integrator_order):
                for l, langevin in enumerate(langevin_order):
                    row = df[
                        (df['mh'] == mh) &
                        (df['canonical'] == canonical) &
                        (df['integrator'] == integrator) &
                        (df['langevin'] == langevin)
                    ]
                    if not row.empty:
                        value = row['value'].values[0]
                        processed_value, is_inf = preprocess_values([value], cap_value)
                        processed_value = processed_value[0]
                        
                        x = i + (j - n_hue/2) * bar_width + (k - n_hatch/2) * (bar_width / n_hatch)
                        bar = ax.bar(
                            x, processed_value, 
                            width=bar_width / n_hatch,
                            color=color_map[canonical],
                            hatch=hatch_map[integrator],
                            edgecolor='black',
                            alpha=alpha_map[langevin],
                            label=f"{canonical}, {integrator}, {langevin}" if i == 0 else ""
                        )
                        
                        # Add infinity marker for capped values
                        if is_inf[0]:
                            ax.text(x, processed_value, '∞', 
                                  ha='center', va='bottom',
                                  fontweight='bold', fontsize=12)
    
    # Customize plot
    ax.set_xticks(np.arange(len(mh_order)))
    ax.set_xticklabels(mh_order)
    ax.set_xlabel('MH Adjustment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Model: {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Create separate legends for each attribute
    legend_elements = []
    
    # Canonical vs Microcanonical legend
    for canonical in canonical_order:
        legend_elements.append(
            Patch(facecolor=color_map[canonical], 
                  label=canonical,
                  edgecolor='black')
        )
    first_legend = ax.legend(handles=legend_elements, 
                           title='canonical (color)',
                           bbox_to_anchor=(1.01, 1), 
                           loc='upper left')
    ax.add_artist(first_legend)
    
    # Integrator legend
    legend_elements = []
    for integrator in integrator_order:
        legend_elements.append(
            Patch(facecolor='white',
                  hatch=hatch_map[integrator],
                  label=integrator,
                  edgecolor='black')
        )
    second_legend = ax.legend(handles=legend_elements,
                            title='integrator (hatch)',
                            bbox_to_anchor=(1.01, 0.7),
                            loc='upper left')
    ax.add_artist(second_legend)
    
    # Langevin legend
    legend_elements = []
    for langevin in langevin_order:
        legend_elements.append(
            Patch(facecolor='gray',
                  alpha=alpha_map[langevin],
                  label=langevin,
                  edgecolor='black')
        )
    ax.legend(handles=legend_elements,
             title='langevin (transparency)',
             bbox_to_anchor=(1.01, 0.4),
             loc='upper left')
    
    # Adjust layout to prevent legend overlap
    plt.subplots_adjust(right=0.85)
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    
    plt.close() 