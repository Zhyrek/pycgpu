#!/usr/bin/env python
"""Simple comparison of CPU vs GPU equilibrium."""

from pycalphad import Database, equilibrium

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Single condition
conditions = {
    'X_TI': 0.5,
    'T': 1000
}

print("Running CPU calculation...")
eq_cpu = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
print(f"CPU Final GM: {eq_cpu.GM.values[0]}")

print("\nRunning GPU calculation...")
eq_gpu = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU Final GM: {eq_gpu.GM.values[0]}")

print(f"\nDifference: {abs(eq_cpu.GM.values[0] - eq_gpu.GM.values[0])}")