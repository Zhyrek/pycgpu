#!/usr/bin/env python
"""Test if skipping advance_state after phase changes fixes GPU convergence."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing GPU equilibrium with phase change fix...")
print("=" * 50)

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

# Run CPU calculation for comparison
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]

print(f"\nTarget X(TI): 0.9000000000")
print(f"CPU X(TI):    {cpu_x_ti:.10f}")
print(f"GPU X(TI):    {gpu_x_ti:.10f}")
print(f"Difference:   {abs(gpu_x_ti - cpu_x_ti):.10f}")

if abs(gpu_x_ti - 0.9) < 1e-6:
    print("\n✅ SUCCESS: GPU now converges correctly!")
else:
    print(f"\n❌ FAILED: GPU still has error of {gpu_x_ti - 0.9:.10f}")