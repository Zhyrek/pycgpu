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

# Show site fractions (Y) for each point
print("\nSite fractions (Y):")
for i in range(calc_result.X.shape[-1]):
    y_nb = calc_result.Y.sel(component='NB', vertex=i).values[0,0,0,0,0]
    y_ti = calc_result.Y.sel(component='TI', vertex=i).values[0,0,0,0,0]
    gm = calc_result.GM.sel(vertex=i).values[0,0,0,0]
    print(f"  Point {i}: Y(NB)={y_nb:.6f}, Y(TI)={y_ti:.6f}, GM={gm:.1f} J/mol")

# Show mole fractions (X) for each point
print("\nMole fractions (X):")
for i in range(calc_result.X.shape[-1]):
    x_nb = calc_result.X.sel(component='NB', vertex=i).values[0,0,0,0]
    x_ti = calc_result.X.sel(component='TI', vertex=i).values[0,0,0,0]
    print(f"  Point {i}: X(NB)={x_nb:.6f}, X(TI)={x_ti:.6f}")

# Check difference between mole fractions
print("\nComposition differences between points:")
for comp in ['NB', 'TI']:
    x0 = calc_result.X.sel(component=comp, vertex=0).values[0,0,0,0]
    x1 = calc_result.X.sel(component=comp, vertex=1).values[0,0,0,0]
    diff = abs(x0 - x1)
    print(f"  |X({comp})_0 - X({comp})_1| = {diff:.6f}")

print(f"\nConsolidation threshold: 1e-4 = {1e-4:.6f}")
print("Should consolidate: ", all(abs(calc_result.X.sel(component=comp, vertex=0).values[0,0,0,0] - 
                                    calc_result.X.sel(component=comp, vertex=1).values[0,0,0,0]) < 1e-4 
                                   for comp in ['NB', 'TI']))