#!/usr/bin/env python
"""Test comprehensive conditions focusing on T=1500K."""

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

# Test a range of compositions at T=1500K
print("Testing GPU behavior at T=1500K across different compositions...")
print("-" * 80)
print(f"{'X(TI)':>6} | {'CPU GM':>15} | {'GPU GM':>15} | {'Difference':>12} | Notes")
print("-" * 80)

x_ti_values = np.linspace(0.1, 0.9, 9)
failures = []

for x_ti in x_ti_values:
    conditions = {
        v.T: 1500,
        v.P: 101325,
        v.X('TI'): x_ti
    }
    
    # Run CPU calculation
    try:
        eq_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = eq_cpu.GM.values.item()
    except Exception as e:
        cpu_gm = np.nan
        print(f"{x_ti:>6.2f} | CPU Error: {e}")
        continue
    
    # Run GPU calculation
    try:
        eq_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = eq_gpu.GM.values.item()
    except Exception as e:
        gpu_gm = np.nan
        print(f"{x_ti:>6.2f} | {cpu_gm:>15.6f} | GPU Error: {e}")
        failures.append((x_ti, str(e)))
        continue
    
    # Compare results
    diff = abs(gpu_gm - cpu_gm) if not np.isnan(gpu_gm) and not np.isnan(cpu_gm) else np.nan
    
    notes = ""
    if gpu_gm == 0.0:
        notes = "GPU returned 0.0!"
        failures.append((x_ti, "GPU returned 0.0"))
    elif np.isnan(gpu_gm):
        notes = "GPU returned NaN"
        failures.append((x_ti, "GPU returned NaN"))
    elif diff > 1.0:
        notes = "Large difference"
        failures.append((x_ti, f"Large difference: {diff:.6e}"))
    
    print(f"{x_ti:>6.2f} | {cpu_gm:>15.6f} | {gpu_gm:>15.6f} | {diff:>12.6e} | {notes}")

print("-" * 80)

if failures:
    print(f"\nFound {len(failures)} failure(s):")
    for x_ti, reason in failures:
        print(f"  X(TI)={x_ti:.2f}: {reason}")
else:
    print("\nNo failures found - GPU is working correctly at T=1500K")