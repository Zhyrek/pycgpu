#!/usr/bin/env python3
"""Test what starting_point returns"""

from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.lower_convex_hull import lower_convex_hull
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Starting Point Analysis ===\n")

# First, run calculate with two points
calc_result = calculate(db, comps, 'BCC_A2', T=300, P=101325, N=1,
                       points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print(f"1. Calculate result:")
print(f"   Shape: {calc_result.GM.shape}")
print(f"   Number of points: {calc_result.GM.shape[-1]}")
for i in range(calc_result.GM.shape[-1]):
    gm = calc_result.GM.values.flat[i]
    print(f"   Point {i}: GM = {gm:.1f} J/mol")

# Now let's see what lower_convex_hull does
print("\n2. Lower convex hull:")
hull_result = lower_convex_hull(calc_result, {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}, ['BCC_A2'])

print(f"   Hull result shape: {hull_result.GM.shape}")
print(f"   Hull Phase shape: {hull_result.Phase.shape}")
print(f"   Hull NP shape: {hull_result.NP.shape}")
print(f"   Hull Phase values: {hull_result.Phase.values}")
print(f"   Hull NP values: {hull_result.NP.values}")

# Check how many phases are active in hull result
active_count = 0
for i in range(hull_result.NP.shape[-1]):
    np_val = hull_result.NP.values.flat[i]
    phase = hull_result.Phase.values.flat[i]
    if np_val > 1e-6 and phase != '':
        active_count += 1
        print(f"   Active phase {i}: {phase}, NP = {np_val:.6f}")

print(f"\n   Total active phases from hull: {active_count}")

# Compare CPU equilibrium
print("\n3. CPU equilibrium for comparison:")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}})

print(f"   CPU GM: {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"   CPU active phases: {[p for p in eq_cpu.Phase.values.flat if p != '']}")
print(f"   CPU phase amounts: {eq_cpu.NP.values[eq_cpu.NP.values > 1e-6]}")