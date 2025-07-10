#!/usr/bin/env python3
"""Check starting_point output in detail"""

from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.workspace import Workspace
from collections import OrderedDict
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Starting Point Detail Test ===\n")

# Create workspace to get phase records
wks = Workspace(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}})

# Run calculate with two starting points
grid = calculate(db, comps, phases, T=300, P=101325, N=1,
                points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print(f"1. Calculate grid:")
print(f"   GM shape: {grid.GM.shape}")
print(f"   Phase shape: {grid.Phase.shape}")
print(f"   Number of points: {grid.Phase.shape[-1]}")

# Show all phases in grid
for i in range(grid.Phase.shape[-1]):
    phase = grid.Phase.values.flat[i]
    gm = grid.GM.values.flat[i]
    print(f"   Grid point {i}: Phase='{phase}', GM={gm:.1f}")

# Run starting_point
state_variables = sorted([v.N, v.P, v.T], key=str)
unitless_conds = OrderedDict()
for key, val in conditions.items():
    if key in state_variables:
        unitless_conds[key] = np.array([val])

print(f"\n2. Running starting_point with verbose=True:")
sp_result = starting_point(unitless_conds, state_variables, 
                          wks.phase_record_factory, grid, verbose=True)

print(f"\n3. Starting point result:")
print(f"   Phase shape: {sp_result.Phase.shape}")
print(f"   NP shape: {sp_result.NP.shape}")
print(f"   Phase values: {sp_result.Phase.values}")
print(f"   NP values: {sp_result.NP.values}")

# Count non-empty phases
active_count = 0
for i in range(sp_result.Phase.shape[-1]):
    phase = sp_result.Phase.values.flat[i]
    np_val = sp_result.NP.values.flat[i]
    if phase != '' and phase != '_FAKE_' and np_val > 1e-8:
        active_count += 1
        print(f"\n   Active phase {i}: '{phase}', NP={np_val:.6f}")
        # Get site fractions
        if hasattr(sp_result, 'Y'):
            y_vals = sp_result.Y.values[0,0,0,i,:]
            print(f"   Site fractions: {y_vals[:2]}")

print(f"\nTotal active phases from starting_point: {active_count}")