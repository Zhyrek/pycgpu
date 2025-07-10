#!/usr/bin/env python3
"""Test phase consolidation with NbTi.tdb"""

from pycalphad import Database, equilibrium, calculate, variables as v
import numpy as np

# Load the existing NbTi.tdb database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions - T=300K, X(TI)=0.4
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Testing Phase Consolidation with NbTi.tdb ===\n")

# Run CPU equilibrium with two initial points
print("--- CPU Equilibrium ---")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

print(f"\nCPU Result:")
print(f"  GM: {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"  Phases: {eq_cpu.Phase.values}")
print(f"  NP: {eq_cpu.NP.values}")
print(f"  X(NB) in phases: {eq_cpu.X.sel(component='NB').values}")
print(f"  X(TI) in phases: {eq_cpu.X.sel(component='TI').values}")

# Run GPU equilibrium  
print("\n--- GPU Equilibrium ---")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

print(f"\nGPU Result:")
print(f"  GM: {eq_gpu.GM.values[0]:.1f} J/mol")
print(f"  Phases: {eq_gpu.Phase.values}")
print(f"  NP: {eq_gpu.NP.values}")
print(f"  X(NB) in phases: {eq_gpu.X.sel(component='NB').values}")
print(f"  X(TI) in phases: {eq_gpu.X.sel(component='TI').values}")

print(f"\n=== Comparison ===")
print(f"GM Difference: {abs(eq_gpu.GM.values[0] - eq_cpu.GM.values[0]):.1f} J/mol")
print(f"Expected: < 0.001 J/mol")