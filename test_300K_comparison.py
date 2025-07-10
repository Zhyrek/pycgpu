#!/usr/bin/env python3
"""Compare CPU and GPU at T=300K"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions - T=300K
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== T=300K Comparison ===\n")

# Run CPU
print("Running CPU...")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

print(f"\nCPU GM: {float(eq_cpu.GM.values.flat[0]):.1f} J/mol")

# Run GPU
print("\nRunning GPU...")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

print(f"\nGPU GM: {float(eq_gpu.GM.values.flat[0]):.1f} J/mol")
print(f"\nDifference: {abs(float(eq_gpu.GM.values.flat[0]) - float(eq_cpu.GM.values.flat[0])):.1f} J/mol")