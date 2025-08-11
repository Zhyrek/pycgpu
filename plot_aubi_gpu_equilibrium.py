#!/usr/bin/env python
"""Plot Au-Bi phase diagram using GPU equilibrium results colored by phase ID."""

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
    v.X('BI'): (0.05, 0.95, 0.05),  # 0.05 to 0.95 in 0.05 increments
    v.T: (300, 1400, 20),            # 300 to 1400 K in 20K increments
    v.P: 101325
}

# Run GPU equilibrium calculation
print("Running GPU equilibrium calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, 
                         calc_opts={'pdens': 100}, verbose=False)

print(f"Result shape: {result_gpu.dims}")

# Extract data
x_bi = result_gpu.coords['X_BI'].values
temps = result_gpu.coords['T'].values
print(f"X_BI values: {len(x_bi)} points from {x_bi.min():.2f} to {x_bi.max():.2f}")
print(f"Temperature values: {len(temps)} points from {temps.min():.0f}K to {temps.max():.0f}K")

# Get phase information at each point
# The Phase variable contains the stable phases at each condition
phase_data = result_gpu.Phase.values  # Shape: (N, P, T, X_BI, vertex)

# Extract dominant phase at each point
# We'll look at which phase has the highest phase fraction (NP)
np_data = result_gpu.NP.values  # Phase fractions

# Create a phase ID map
phase_names = list(set(phase_data.flatten()) - {''})
phase_names = [p for p in phase_names if p and p != '_FAKE_']
phase_names.sort()
print(f"Found phases in equilibrium: {phase_names}")

# Create phase ID mapping
phase_to_id = {phase: i for i, phase in enumerate(phase_names)}
phase_to_id[''] = -1
phase_to_id['_FAKE_'] = -1

# Create 2D grid for plotting
X, Y = np.meshgrid(x_bi, temps)
phase_ids = np.zeros_like(X)

# Determine dominant phase at each point
for t_idx, t in enumerate(temps):
    for x_idx, x in enumerate(x_bi):
        # Get phases and their amounts at this condition
        phases_here = phase_data[0, 0, t_idx, x_idx, :]
        amounts_here = np_data[0, 0, t_idx, x_idx, :]
        
        # Find dominant phase (highest amount)
        valid_mask = (phases_here != '') & (phases_here != '_FAKE_')
        if np.any(valid_mask):
            max_idx = np.argmax(amounts_here[valid_mask])
            dominant_phase = phases_here[valid_mask][max_idx]
            phase_ids[t_idx, x_idx] = phase_to_id.get(dominant_phase, -1)
        else:
            phase_ids[t_idx, x_idx] = -1

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Phase regions colored by phase ID
# Create custom colormap for phases
colors = ['white', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
n_phases = len(phase_names)
cmap = ListedColormap(colors[:n_phases])
bounds = np.arange(-0.5, n_phases, 1)
norm = BoundaryNorm(bounds, n_phases)

im1 = ax1.pcolormesh(X, Y, phase_ids, cmap=cmap, norm=norm, shading='nearest')
ax1.set_xlabel('Mole Fraction Bi', fontsize=12)
ax1.set_ylabel('Temperature (K)', fontsize=12)
ax1.set_title('Au-Bi Phase Diagram (GPU Equilibrium)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Add colorbar with phase names
cbar1 = plt.colorbar(im1, ax=ax1, ticks=range(n_phases))
cbar1.set_label('Phase', fontsize=12)
cbar1.ax.set_yticklabels(phase_names)

# Plot 2: Number of phases in equilibrium
n_phases_eq = np.zeros_like(X)
for t_idx, t in enumerate(temps):
    for x_idx, x in enumerate(x_bi):
        phases_here = phase_data[0, 0, t_idx, x_idx, :]
        amounts_here = np_data[0, 0, t_idx, x_idx, :]
        # Count phases with significant amount (> 0.01)
        valid_mask = (phases_here != '') & (phases_here != '_FAKE_') & (amounts_here > 0.01)
        n_phases_eq[t_idx, x_idx] = np.sum(valid_mask)

im2 = ax2.pcolormesh(X, Y, n_phases_eq, cmap='viridis', shading='nearest', vmin=0, vmax=3)
ax2.set_xlabel('Mole Fraction Bi', fontsize=12)
ax2.set_ylabel('Temperature (K)', fontsize=12)
ax2.set_title('Number of Phases in Equilibrium', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Number of Phases', fontsize=12)

# Add phase boundaries (where phase ID changes)
# Detect boundaries
phase_boundaries_x = []
phase_boundaries_y = []

for t_idx in range(len(temps)-1):
    for x_idx in range(len(x_bi)-1):
        if phase_ids[t_idx, x_idx] != phase_ids[t_idx, x_idx+1]:
            # Vertical boundary
            phase_boundaries_x.append(x_bi[x_idx] + 0.005)
            phase_boundaries_y.append(temps[t_idx])
        if phase_ids[t_idx, x_idx] != phase_ids[t_idx+1, x_idx]:
            # Horizontal boundary
            phase_boundaries_x.append(x_bi[x_idx])
            phase_boundaries_y.append(temps[t_idx] + 10)

# Plot boundaries as scatter points on first plot
if phase_boundaries_x:
    ax1.scatter(phase_boundaries_x, phase_boundaries_y, c='black', s=0.5, alpha=0.5)

plt.suptitle('Au-Bi System: GPU Equilibrium Analysis', fontsize=16)
plt.tight_layout()

# Save figure
plt.savefig('aubi_phase_diagram_gpu.png', dpi=150)
print("Saved GPU phase diagram to aubi_phase_diagram_gpu.png")

# Print statistics
print("\nPhase distribution statistics:")
for phase in phase_names:
    phase_id = phase_to_id[phase]
    count = np.sum(phase_ids == phase_id)
    percentage = 100 * count / phase_ids.size
    print(f"  {phase}: {count} points ({percentage:.1f}%)")

unique_n_phases = np.unique(n_phases_eq)
print(f"\nNumber of phases in equilibrium:")
for n in unique_n_phases:
    count = np.sum(n_phases_eq == n)
    percentage = 100 * count / n_phases_eq.size
    print(f"  {int(n)} phases: {count} points ({percentage:.1f}%)")

plt.close()