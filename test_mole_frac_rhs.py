#!/usr/bin/env python3
"""Test mole fraction constraint RHS calculation"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Testing Mole Fraction Constraint RHS ===\n")

# Run CPU 
print("CPU calculation:")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}})
print(f"CPU GM: {float(eq_cpu.GM.values.flat[0]):.1f} J/mol")

# Run GPU
print("\nGPU calculation:")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True)
print(f"GPU GM: {float(eq_gpu.GM.values.flat[0]):.1f} J/mol")

print(f"\nDifference: {abs(float(eq_gpu.GM.values.flat[0]) - float(eq_cpu.GM.values.flat[0])):.1f} J/mol")

# The key diagnostic is to see if GPU mole fraction RHS matches CPU
# CPU shows: RHS = -0.497686
# GPU was showing: RHS = -0.299434 (incorrect)
# After fix, GPU should show: RHS = -0.497686