#!/usr/bin/env python
"""Test GPU damping fix after consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING GPU DAMPING FIX")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nExpected behavior with damping:")
print("- Phases consolidate to X(TI) = 0.903147")
print("- Damping prevents updates for 2 iterations")
print("- X(TI) should remain 0.903147 for iterations 1-2")
print("- Then update to achieve X(TI) = 0.900000")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    gpu_overall = sum(np * x for np, x in zip(gpu_np, gpu_x_ti) if np > 1e-12)
    print(f"\nGPU final result: X(TI) = {gpu_overall:.10f}")
    
    if abs(gpu_overall - 0.9) < 1e-6:
        print("SUCCESS: GPU now achieves X(TI) = 0.900000!")
        print("The damping fix worked!")
    else:
        print(f"Still not exact: difference = {abs(gpu_overall - 0.9):.10f}")
        
except Exception as e:
    print(f"GPU failed: {e}")

print("\n" + "="*60)
print("Looking for damping messages in debug output...")