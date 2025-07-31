#!/usr/bin/env python
"""Test GPU behavior at T=1500K with multiple conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test with conditions that include T=1500K
conditions = {
    v.T: [1400, 1450, 1500, 1550, 1600],
    v.P: 101325,
    v.X('TI'): 0.5
}

print("Testing GPU behavior at T=1500K with multiple conditions...")
print("Conditions:", conditions)

# Run CPU calculation
print("\nRunning CPU calculation...")
eq_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
print(f"CPU GM shape: {eq_cpu.GM.shape}")
print(f"CPU GM values: {eq_cpu.GM.values}")

# Run GPU calculation  
print("\nRunning GPU calculation...")
eq_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
print(f"GPU GM shape: {eq_gpu.GM.shape}")
print(f"GPU GM values: {eq_gpu.GM.values}")

# Compare specific values
print("\nDetailed comparison:")
print("-" * 70)
print(f"{'T(K)':>6} | {'CPU GM':>15} | {'GPU GM':>15} | {'Difference':>12} | Notes")
print("-" * 70)

temps = eq_cpu.coords['T'].values
for i, T in enumerate(temps):
    cpu_val = eq_cpu.GM.values.flat[i]
    gpu_val = eq_gpu.GM.values.flat[i]
    diff = abs(gpu_val - cpu_val)
    
    notes = ""
    if gpu_val == 0.0:
        notes = "GPU returned 0.0!"
    elif np.isnan(gpu_val):
        notes = "GPU returned NaN"
    elif diff > 1.0:
        notes = "Large difference"
    
    print(f"{T:>6.0f} | {cpu_val:>15.6f} | {gpu_val:>15.6f} | {diff:>12.6e} | {notes}")

print("-" * 70)