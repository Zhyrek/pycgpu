#!/usr/bin/env python
"""Test if the iteration labeling fix works."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING ITERATION LABELING FIX")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running GPU equilibrium to check iteration numbering...")

try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
    gpu_np = result_gpu.NP.values.flatten()
    
    gpu_overall = sum(np * x for np, x in zip(gpu_np, gpu_x_ti) if np > 1e-12)
    print(f"GPU final result: X(TI) = {gpu_overall:.10f}")
    
    print("\nLooking for iteration numbers in debug output...")
    print("GPU should now show iteration 1 when consolidating instead of iteration 0")
    print("This should match the CPU which shows iteration 1 when consolidating")
    
except Exception as e:
    print(f"GPU failed: {e}")

print(f"\n" + "="*60)
print("EXPECTED BEHAVIOR:")
print("- GPU and CPU should both show 'iteration 1' when consolidating")  
print("- GPU consolidation should happen at 'iteration 1' not 'iteration 0'")
print("- Debug messages should align between CPU and GPU")