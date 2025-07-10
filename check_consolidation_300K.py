#!/usr/bin/env python3
"""Check what happens at T=300K with X(TI)=0.4"""

from pycalphad import Database, equilibrium, calculate, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA'] 
phases = ['BCC_A2']

# Set conditions - this is from the original GPU test
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Checking equilibrium at T=300K, X(TI)=0.4 ===\n")

# First check what calculate gives us with default points
print("1. Calculate with default grid:")
calc_default = calculate(db, comps, 'BCC_A2', T=300, P=101325, N=1)
print(f"   Number of points: {calc_default.X.shape[-1]}")

# Now run equilibrium to see what happens
print("\n2. CPU Equilibrium with default grid:")
eq_cpu = equilibrium(db, comps, phases, conditions, verbose=True)
print(f"   CPU GM: {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"   CPU phases present: {[p for p in eq_cpu.Phase.values.flat if p != '']}")
print(f"   CPU NP values: {eq_cpu.NP.values[eq_cpu.NP.values > 1e-6]}")

# Check with specific starting points from the original test
print("\n3. CPU Equilibrium with points [[0.6, 0.4], [0.5, 0.5]]:")
eq_cpu2 = equilibrium(db, comps, phases, conditions,
                     calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}}, 
                     verbose=True)
print(f"   CPU GM: {eq_cpu2.GM.values[0]:.1f} J/mol")

# Now GPU
print("\n4. GPU Equilibrium with points [[0.6, 0.4], [0.5, 0.5]]:")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)
print(f"   GPU GM: {eq_gpu.GM.values[0]:.1f} J/mol")

print(f"\nDifference: {abs(eq_gpu.GM.values[0] - eq_cpu2.GM.values[0]):.1f} J/mol")