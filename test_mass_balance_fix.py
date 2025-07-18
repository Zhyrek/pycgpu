#!/usr/bin/env python
"""Test GPU mass balance fix when phases are removed."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING GPU MASS BALANCE FIX")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nExpected behavior:")
print("- Phase 1 is removed (small amount)")
print("- GPU adjusts remaining phase to X(TI) = 0.9")
print("- This maintains mass balance")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    gpu_overall = sum(np * x for np, x in zip(gpu_np, gpu_x_ti) if np > 1e-12)
    print(f"\nGPU final result: X(TI) = {gpu_overall:.10f}")
    
    if abs(gpu_overall - 0.9) < 1e-6:
        print("SUCCESS: GPU now achieves X(TI) = 0.900000!")
        print("The mass balance fix worked!")
    else:
        print(f"Error = {abs(gpu_overall - 0.9):.10f}")
        
except Exception as e:
    print(f"GPU failed: {e}")