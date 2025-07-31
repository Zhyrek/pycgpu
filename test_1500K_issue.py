#!/usr/bin/env python
"""Test specific issue with GPU returning 0.0 at T=1500K."""

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

# Test conditions around T=1500K
test_conditions = [
    {'T': 1400, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1450, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1500, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1550, 'P': 101325, 'X(TI)': 0.5},
    {'T': 1600, 'P': 101325, 'X(TI)': 0.5},
]

print("Testing GPU behavior around T=1500K...")
print("-" * 70)
print(f"{'T(K)':>6} | {'CPU GM':>15} | {'GPU GM':>15} | {'Difference':>12} | Notes")
print("-" * 70)

for conditions in test_conditions:
    T = conditions['T']
    
    # Run CPU calculation
    try:
        eq_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = eq_cpu.GM.values.item()
    except Exception as e:
        cpu_gm = np.nan
        print(f"{T:>6} | CPU Error: {e}")
        continue
    
    # Run GPU calculation
    try:
        eq_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = eq_gpu.GM.values.item()
    except Exception as e:
        gpu_gm = np.nan
        print(f"{T:>6} | {'N/A':>15} | GPU Error: {e}")
        continue
    
    # Compare results
    diff = abs(gpu_gm - cpu_gm) if not np.isnan(gpu_gm) and not np.isnan(cpu_gm) else np.nan
    
    notes = ""
    if gpu_gm == 0.0:
        notes = "GPU returned 0.0!"
    elif np.isnan(gpu_gm):
        notes = "GPU returned NaN"
    elif diff > 1.0:
        notes = "Large difference"
    
    print(f"{T:>6} | {cpu_gm:>15.6f} | {gpu_gm:>15.6f} | {diff:>12.6e} | {notes}")

print("-" * 70)