#!/usr/bin/env python
"""Test GPU formula units fix."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running GPU calculation with formula units fix...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"\nGPU X(TI): {gpu_x_ti:.10f}")
print(f"Target X(TI): 0.9000000000")
print(f"Absolute error: {abs(gpu_x_ti - 0.9):.10f}")

if abs(gpu_x_ti - 0.9) < 1e-6:
    print("\n✓ SUCCESS: GPU now converges correctly!")
else:
    print(f"\n✗ FAILED: GPU still produces X(TI) = {gpu_x_ti:.10f}")