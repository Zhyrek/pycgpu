#!/usr/bin/env python3
"""Debug starting point calculation"""

from pycalphad import Database, calculate, variables as v
from pycalphad.core.starting_point import starting_point
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Starting Point Debug ===\n")

# Run calculate with specific starting points
print("1. Calculate with two points:")
calc_result = calculate(db, comps, 'BCC_A2', T=300, P=101325, N=1,
                       points={'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]})

print(f"   Calculate result shape: {calc_result.GM.shape}")
print(f"   Number of points: {calc_result.GM.shape[-1]}")

# Show the points
for i in range(calc_result.GM.shape[-1]):
    gm = calc_result.GM.values.flat[i]
    print(f"   Point {i}: GM = {gm:.1f} J/mol")

# Run starting_point
print("\n2. Starting point calculation:")
sp_result = starting_point(calc_result, conditions, phases)

print(f"   Starting point NP shape: {sp_result.NP.shape}")
print(f"   Starting point Phase shape: {sp_result.Phase.shape}")
print(f"   Starting point GM shape: {sp_result.GM.shape}")

# Check what phases are present
print("\n3. Phases in starting point:")
phase_values = sp_result.Phase.values.flat
np_values = sp_result.NP.values.flat
for i, (phase, np_val) in enumerate(zip(phase_values, np_values)):
    if phase != '' and np_val > 1e-6:
        print(f"   Phase {i}: {phase}, NP = {np_val:.6f}")

print(f"\n4. Starting point GM: {sp_result.GM.values[0]:.1f} J/mol")