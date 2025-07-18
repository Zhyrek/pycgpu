#!/usr/bin/env python
"""Test to see detailed behavior during phase consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing phase consolidation behavior...")
print("=" * 50)

# Run GPU calculation with verbose output
print("\n--- GPU Calculation (verbose) ---")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"\n\nFinal GPU X(TI): {gpu_x_ti:.10f}")
print(f"Target X(TI):    0.9000000000")
print(f"Error:           {gpu_x_ti - 0.9:.10f}")

if abs(gpu_x_ti - 0.9) < 1e-6:
    print("\n✅ SUCCESS: GPU converges correctly!")
else:
    print(f"\n❌ FAILED: GPU error = {gpu_x_ti - 0.9:.10f}")