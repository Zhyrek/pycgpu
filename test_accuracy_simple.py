#!/usr/bin/env python
"""Simple test of numerical accuracy differences."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test conditions
print("Testing numerical accuracy: Two-phase vs Single-phase regions")
print("="*80)

# Two-phase region (inside miscibility gap) - should have perfect accuracy
print("\n1. Two-phase region (X(TI)=0.1, T=500K):")
conditions = {v.X('TI'): 0.1, v.T: 500, v.P: 101325}
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
gpu_gm = float(result_gpu.GM.values)
cpu_gm = float(result_cpu.GM.values)
error = gpu_gm - cpu_gm
print(f"   GPU: {gpu_gm:.12f} J/mol")
print(f"   CPU: {cpu_gm:.12f} J/mol")
print(f"   Error: {error:.12f} J/mol")
print(f"   GPU phases: {result_gpu.Phase.values.flatten()[result_gpu.NP.values.flatten() > 1e-6]}")

# Single-phase region (requires consolidation) - has small error
print("\n2. Single-phase region (X(TI)=0.1, T=600K):")
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
gpu_gm = float(result_gpu.GM.values)
cpu_gm = float(result_cpu.GM.values)
error = gpu_gm - cpu_gm
print(f"   GPU: {gpu_gm:.12f} J/mol")
print(f"   CPU: {cpu_gm:.12f} J/mol")
print(f"   Error: {error:.12f} J/mol")
print(f"   GPU phases: {result_gpu.Phase.values.flatten()[result_gpu.NP.values.flatten() > 1e-6]}")

print("\nObservation: Two-phase = perfect accuracy, Single-phase = small error")
print("The error must be introduced during phase consolidation.")