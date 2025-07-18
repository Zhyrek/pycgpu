#!/usr/bin/env python
"""Test that the constraint fix works for multiple values."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing GPU constraint fix for multiple X(TI) values...")
print("Expected: All should return exactly the prescribed value")
print("=" * 60)

test_values = [0.005, 0.010, 0.020]

for x_ti_target in test_values:
    print(f"\nTesting X(TI) = {x_ti_target:.3f}")
    print("-" * 30)
    
    conditions = {v.X('TI'): x_ti_target, v.T: 1000, v.P: 101325}
    
    # Test GPU
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    x_ti_gpu = result_gpu.X.sel(component='TI').values.flatten()[0]
    
    # Test CPU for comparison  
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    x_ti_cpu = result_cpu.X.sel(component='TI').values.flatten()[0]
    
    gpu_error = abs(x_ti_gpu - x_ti_target)
    cpu_error = abs(x_ti_cpu - x_ti_target)
    
    print(f"  Target:    {x_ti_target:.10f}")
    print(f"  CPU:       {x_ti_cpu:.10f} (error = {cpu_error:.2e})")
    print(f"  GPU:       {x_ti_gpu:.10f} (error = {gpu_error:.2e})")
    
    if gpu_error < 1e-10:
        print(f"  ✓ GPU constraint satisfied exactly")
    elif gpu_error < 1e-6:
        print(f"  ✓ GPU constraint satisfied within tolerance") 
    else:
        print(f"  ✗ GPU constraint NOT satisfied")
        
    if abs(x_ti_gpu - x_ti_cpu) < 1e-10:
        print(f"  ✓ GPU matches CPU exactly")
    elif abs(x_ti_gpu - x_ti_cpu) < 1e-6:
        print(f"  ✓ GPU matches CPU within tolerance")
    else:
        print(f"  ✗ GPU does NOT match CPU")

print("\n" + "=" * 60)
print("Summary: GPU constraint handling fix test complete")