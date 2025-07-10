#!/usr/bin/env python3
"""Debug CPU starting point in detail"""

from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.starting_point import starting_point
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== CPU Starting Point Debug ===\n")

# Create workspace like equilibrium does
wks = Workspace(db, comps, phases, conditions, 
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}})

print(f"Workspace components: {wks.components}")
print(f"Workspace phases: {wks.phases}")

# Run calculate
grid = calculate(db, comps, phases, T=conditions[v.T], P=conditions[v.P], N=conditions[v.N],
                points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]},
                phase_records=wks.phase_record_factory)

print(f"\nCalculate result GM shape: {grid.GM.shape}")
print("Calculate results:")
for i in range(grid.GM.shape[-1]):
    gm = grid.GM.values.flat[i]
    print(f"  Point {i}: GM = {gm:.1f} J/mol")

# Get state variables
state_variables = sorted([v.N, v.P, v.T], key=str)

# Call starting_point
unitless_conds = wks.get_unitless_conditions(state_variables)
print(f"\nCalling starting_point with conditions: {unitless_conds}")

sp_result = starting_point(unitless_conds, state_variables, 
                          wks.phase_record_factory, grid, 
                          verbose=True)

print(f"\nStarting point result:")
print(f"  NP shape: {sp_result.NP.shape}")
print(f"  Phase shape: {sp_result.Phase.shape}")
print(f"  GM: {sp_result.GM.values[0]:.1f} J/mol")

# Count active phases
active_phases = 0
for i in range(sp_result.NP.shape[-1]):
    np_val = sp_result.NP.values.flat[i]
    phase = sp_result.Phase.values.flat[i]
    if np_val > 1e-6 and phase != '':
        active_phases += 1
        print(f"  Active phase {i}: {phase}, NP = {np_val:.6f}")

print(f"\nTotal active phases in starting point: {active_phases}")