#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== Tracing Phase Consolidation ===")
print("Testing with T=1000K, X(TI)=0.01\n")

conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run CPU
print("--- CPU Calculation ---")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    print(f"✓ CPU completed: X(TI) = {cpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"✗ CPU failed: {e}")

# Run GPU  
print("\n--- GPU Calculation ---")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print(f"✓ GPU completed: X(TI) = {gpu_result.X.sel(component='TI').values.flatten()[0]:.6f}")
except Exception as e:
    print(f"✗ GPU failed: {e}")
