#!/usr/bin/env python3
"""Compare initial values between CPU and GPU"""
import os
os.environ['PYCALPHAD_DEBUG'] = '0'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.calculate import calculate
from pycalphad.core.lower_convex_hull import lower_convex_hull

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Comparing Initial Values (before equilibrium solver) ===")

# Run calculate and lower_convex_hull (same for both CPU and GPU)
calc_result = calculate(db, comps, phases, N=1, P=101325, T=1000, 
                       model=None, points={'BCC_A2': 50, 'LIQUID': 50})

# Apply conditions
conditions_no_N = {k: val for k, val in conditions.items() if k != v.N}
hull_result = lower_convex_hull(calc_result, conditions_no_N)

print("\nInitial hull result (same for CPU and GPU):")
print(f"Number of phases: {len(hull_result.Phase.values[hull_result.Phase.values != ''])}")

# Get initial values
phases_in_hull = hull_result.Phase.values[hull_result.Phase.values != '']
print(f"Phases: {phases_in_hull}")

# Site fractions
y_values = hull_result.Y.values
print("\nInitial site fractions:")
for i, phase in enumerate(phases_in_hull):
    if phase:
        y = y_values[i]
        # For BCC_A2, we have 2 site fractions (NB, TI)
        print(f"  {phase}: Y(NB)={y[0]:.6f}, Y(TI)={y[1]:.6f}")

# Phase amounts
np_values = hull_result.NP.values
print("\nInitial phase amounts:")
for i, phase in enumerate(phases_in_hull):
    if phase:
        print(f"  {phase}: NP={np_values[i]:.6f}")

# Energies
gm_values = hull_result.GM.values
print("\nInitial phase energies:")
for i, phase in enumerate(phases_in_hull):
    if phase:
        print(f"  {phase}: GM={gm_values[i]:.1f} J/mol")

# Chemical potentials from initial calculation
print("\nInitial chemical potentials:")
# These are computed from the hull
mu_nb = float(hull_result.MU.sel(component='NB').values[0])
mu_ti = float(hull_result.MU.sel(component='TI').values[0])
print(f"  μ(NB) = {mu_nb:.1f} J/mol")
print(f"  μ(TI) = {mu_ti:.1f} J/mol")