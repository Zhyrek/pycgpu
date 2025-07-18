#!/usr/bin/env python
"""Test if the SVD tolerance fix resolves the GPU solver issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING SVD TOLERANCE FIX")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nExpected behavior after fix:")
print("- GPU SVD solver should use rcond=1e-15 instead of 1e-10")
print("- This allows handling of poorly conditioned systems")
print("- GPU should produce larger corrections after consolidation")
print("- Final result should be X(TI) = 0.900000")

print("\nRunning GPU equilibrium calculation...")
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
    print(f"\nGPU result: X(TI) = {gpu_x_ti:.10f}")
    
    error = abs(gpu_x_ti - 0.9)
    print(f"Error from target: {error:.10f}")
    
    if error < 1e-6:
        print("\nSUCCESS! GPU now achieves the correct composition.")
        print("The solver tolerance fix resolved the issue.")
    else:
        print(f"\nStill incorrect. GPU gives {gpu_x_ti:.6f} instead of 0.900000")
        print("The tolerance fix may not be sufficient.")
        
except Exception as e:
    print(f"\nGPU failed with error: {e}")

print("\nRunning CPU equilibrium for comparison...")
try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
    print(f"CPU result: X(TI) = {cpu_x_ti:.10f}")
except Exception as e:
    print(f"CPU failed: {e}")