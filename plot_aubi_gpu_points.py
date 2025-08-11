#!/usr/bin/env python
"""Plot Au-Bi phase diagram using GPU equilibrium results with individual points from phase compositions."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = list(dbf.phases.keys())

print(f"Computing Au-Bi equilibrium with GPU for phases: {phases}")

# Define conditions - using ranges for GPU calculation
conditions = {
    v.X('BI'): (0.03, 0.97, 0.03),  # 0.03 to 0.97 in 0.03 increments
    v.T: (300, 1400, 30),            # 300 to 1400 K in 30K increments
    v.P: 101325
}

# Run GPU equilibrium calculation
print("Running GPU equilibrium calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, 
                         calc_opts={'pdens': 100}, verbose=False)

print(f"Result shape: {result_gpu.dims}")

# Extract data
x_bi_input = result_gpu.coords['X_BI'].values
temps_input = result_gpu.coords['T'].values
print(f"Input X_BI values: {len(x_bi_input)} points from {x_bi_input.min():.2f} to {x_bi_input.max():.2f}")
print(f"Input Temperature values: {len(temps_input)} points from {temps_input.min():.0f}K to {temps_input.max():.0f}K")

# Get phase information at each point
phase_data = result_gpu.Phase.values  # Shape: (N, P, T, X_BI, vertex)
np_data = result_gpu.NP.values  # Phase fractions
x_data = result_gpu.X.values  # Compositions (N, P, T, X_BI, vertex, component)

# Create phase ID map
phase_names = list(set(phase_data.flatten()) - {''})
phase_names = [p for p in phase_names if p and p != '_FAKE_']
phase_names.sort()
print(f"Found phases in equilibrium: {phase_names}")

# Create phase ID mapping with distinct colors
phase_to_id = {phase: i for i, phase in enumerate(phase_names)}
phase_to_id[''] = -1
phase_to_id['_FAKE_'] = -1

# Create color map for phases
colors_map = {
    'LIQUID': 'red',
    'FCC_A1': 'blue', 
    'RHOMBOHEDRAL_A7': 'green',
    'AU2BI_C15': 'orange',
    'BCC_A2': 'purple',
    'HCP_A3': 'brown'
}

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Collect all points for plotting
plot_data = {phase: {'x': [], 'y': [], 'alpha': []} for phase in phase_names}

# Process each condition and extract equilibrium compositions
for t_idx, t in enumerate(temps_input):
    for x_idx, x_input in enumerate(x_bi_input):
        # Get phases and their amounts at this condition
        phases_here = phase_data[0, 0, t_idx, x_idx, :]
        amounts_here = np_data[0, 0, t_idx, x_idx, :]
        comps_here = x_data[0, 0, t_idx, x_idx, :, :]  # Shape: (vertex, component)
        
        # For each phase vertex with significant amount
        for vertex_idx in range(len(phases_here)):
            phase_name = phases_here[vertex_idx]
            amount = amounts_here[vertex_idx]
            
            if phase_name and phase_name not in ['', '_FAKE_'] and amount > 0.001:
                # Get the equilibrium composition for this phase
                # Component 0 is AU, component 1 is BI
                x_bi_eq = comps_here[vertex_idx, 1]  # BI composition
                
                # Store the point
                plot_data[phase_name]['x'].append(x_bi_eq)
                plot_data[phase_name]['y'].append(t)
                # Use amount as alpha (transparency) - scale to 0.3-1.0 range
                plot_data[phase_name]['alpha'].append(0.3 + 0.7 * amount)

# Plot points for each phase
for phase in phase_names:
    if plot_data[phase]['x']:
        # Convert to numpy arrays
        x_vals = np.array(plot_data[phase]['x'])
        y_vals = np.array(plot_data[phase]['y'])
        alphas = np.array(plot_data[phase]['alpha'])
        
        # Plot with varying alpha based on phase fraction
        # Use scatter plot with individual alpha values
        color = colors_map.get(phase, 'gray')
        
        # Plot each point individually with its alpha
        for i in range(len(x_vals)):
            ax.scatter(x_vals[i], y_vals[i], c=color, s=10, 
                      alpha=alphas[i], edgecolors='none')
        
        # Add one point to legend with full opacity
        ax.scatter([], [], c=color, s=50, alpha=1.0, label=phase, edgecolors='black', linewidth=0.5)

# Add tie lines for two-phase regions
print("\nDetecting two-phase regions for tie lines...")
tie_line_count = 0
for t_idx, t in enumerate(temps_input[::3]):  # Sample every 3rd temperature
    t_actual_idx = t_idx * 3
    if t_actual_idx >= len(temps_input):
        continue
        
    for x_idx in range(len(x_bi_input)):
        phases_here = phase_data[0, 0, t_actual_idx, x_idx, :]
        amounts_here = np_data[0, 0, t_actual_idx, x_idx, :]
        comps_here = x_data[0, 0, t_actual_idx, x_idx, :, :]
        
        # Find phases with significant amounts
        valid_phases = []
        valid_comps = []
        for vertex_idx in range(len(phases_here)):
            if phases_here[vertex_idx] not in ['', '_FAKE_'] and amounts_here[vertex_idx] > 0.01:
                valid_phases.append(phases_here[vertex_idx])
                valid_comps.append(comps_here[vertex_idx, 1])  # BI composition
        
        # If exactly 2 phases, draw a tie line
        if len(valid_phases) == 2:
            ax.plot(valid_comps, [t, t], 'k-', alpha=0.2, linewidth=0.5)
            tie_line_count += 1

print(f"Added {tie_line_count} tie lines")

# Customize plot
ax.set_xlabel('Mole Fraction Bi (Equilibrium)', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('Au-Bi Phase Diagram - GPU Equilibrium Compositions', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(300, 1400)

# Add legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Add text annotation
ax.text(0.02, 1350, 'Point size/opacity ~ phase fraction', fontsize=9, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Save figure
plt.tight_layout()
plt.savefig('aubi_phase_diagram_gpu_points.png', dpi=150)
print("Saved GPU phase diagram with composition points to aubi_phase_diagram_gpu_points.png")

# Print statistics
print("\nPhase point statistics:")
for phase in phase_names:
    n_points = len(plot_data[phase]['x'])
    if n_points > 0:
        x_range = (min(plot_data[phase]['x']), max(plot_data[phase]['x']))
        t_range = (min(plot_data[phase]['y']), max(plot_data[phase]['y']))
        print(f"  {phase}: {n_points} points")
        print(f"    X(BI) range: {x_range[0]:.3f} to {x_range[1]:.3f}")
        print(f"    T range: {t_range[0]:.0f}K to {t_range[1]:.0f}K")

plt.close()