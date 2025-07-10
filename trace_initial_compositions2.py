#!/usr/bin/env python3
"""Trace initial phase compositions"""

from pycalphad import Database, calculate, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Calculate with two starting points
print("=== Initial Calculate Results ===\n")
calc_result = calculate(db, comps, 'BCC_A2', T=300, P=101325, N=1,
                       points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print(f"Number of calculated points: {calc_result.X.shape[-1]}")
print(f"X dimensions: {calc_result.X.dims}")
print(f"X shape: {calc_result.X.shape}")

# Get compositions directly from X array
x_values = calc_result.X.values[0, 0, 0, :]  # Shape should be (n_components, n_points)
print(f"\nX values shape: {x_values.shape}")

# The component dimension is the first axis
print("\nMole fractions for each point:")
for point_idx in range(x_values.shape[1]):
    print(f"\n  Point {point_idx}:")
    # X has components in order [NB, TI]
    x_nb = x_values[0, point_idx]
    x_ti = x_values[1, point_idx]
    print(f"    X(NB) = {x_nb:.6f}")
    print(f"    X(TI) = {x_ti:.6f}")
    
# Check differences
print("\nComposition differences between points:")
diff_nb = abs(x_values[0, 0] - x_values[0, 1])
diff_ti = abs(x_values[1, 0] - x_values[1, 1]) 
print(f"  |X(NB)_0 - X(NB)_1| = {diff_nb:.6f}")
print(f"  |X(TI)_0 - X(TI)_1| = {diff_ti:.6f}")
print(f"  Max difference: {max(diff_nb, diff_ti):.6f}")

print(f"\nConsolidation threshold: 1e-4 = {1e-4:.6f}")
print(f"Should consolidate: {max(diff_nb, diff_ti) < 1e-4}")

# Also check site fractions
y_values = calc_result.Y.values[0, 0, 0, :, :, :]  # Shape should be (n_sublattices, n_species, n_points)
print(f"\nY values shape: {y_values.shape}")

if y_values.ndim >= 3:
    for point_idx in range(y_values.shape[-1]):
        print(f"\nPoint {point_idx} site fractions:")
        # For BCC_A2 with 2 sublattices, first sublattice has NB,TI
        y_nb = y_values[0, 0, point_idx] if y_values.shape[0] > 0 and y_values.shape[1] > 0 else 0
        y_ti = y_values[0, 1, point_idx] if y_values.shape[0] > 0 and y_values.shape[1] > 1 else 0
        print(f"  Sublattice 0: Y(NB)={y_nb:.6f}, Y(TI)={y_ti:.6f}")