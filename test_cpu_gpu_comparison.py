#!/usr/bin/env python
"""Test CPU vs GPU comparison with the reverted code."""

import os
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import numpy as np

# Create test case
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("=== CPU vs GPU Comparison Test (Reverted Version) ===")
print("Testing with T=1000K, X(TI)=0.01")

# Test conditions that previously showed divergence
conditions = {v.X('TI'): 0.01, v.T: 1000, v.P: 101325}

# Run CPU calculation
print("\nRunning CPU calculation...")
try:
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
    print("✓ CPU calculation completed successfully")
    print(f"  Result shape: {cpu_result.X.shape}")
    print(f"  Phases present: {cpu_result.Phase.values}")
    print(f"  X(TI) values: {cpu_result.X.sel(component='TI').values}")
    cpu_success = True
except Exception as e:
    print(f"✗ CPU calculation failed: {e}")
    cpu_success = False

# Run GPU calculation
print("\nRunning GPU calculation...")
try:
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
    print("✓ GPU calculation completed successfully")
    print(f"  Result shape: {gpu_result.X.shape}")
    print(f"  Phases present: {gpu_result.Phase.values}")
    print(f"  X(TI) values: {gpu_result.X.sel(component='TI').values}")
    gpu_success = True
except Exception as e:
    print(f"✗ GPU calculation failed: {e}")
    print(f"  Error type: {type(e).__name__}")
    if "nvcc" in str(e):
        print("  This is a compilation error - GPU code failed to compile")
    gpu_success = False

# Compare results if both succeeded
if cpu_success and gpu_success:
    print("\n=== Comparison Results ===")
    
    # Compare phase amounts
    cpu_phases = cpu_result.Phase.values.flatten()
    gpu_phases = gpu_result.Phase.values.flatten()
    
    print(f"\nPhases:")
    print(f"  CPU: {cpu_phases}")
    print(f"  GPU: {gpu_phases}")
    
    # Compare compositions
    cpu_x_ti = cpu_result.X.sel(component='TI').values.flatten()
    gpu_x_ti = gpu_result.X.sel(component='TI').values.flatten()
    
    print(f"\nX(TI) values:")
    print(f"  CPU: {cpu_x_ti}")
    print(f"  GPU: {gpu_x_ti}")
    
    # Filter out NaN values and ensure same length for comparison
    cpu_x_ti_valid = cpu_x_ti[~np.isnan(cpu_x_ti)]
    gpu_x_ti_valid = gpu_x_ti[:len(cpu_x_ti_valid)]
    
    print(f"\nValid X(TI) values for comparison:")
    print(f"  CPU: {cpu_x_ti_valid}")
    print(f"  GPU: {gpu_x_ti_valid}")
    
    # Calculate differences
    x_diff = np.abs(cpu_x_ti_valid - gpu_x_ti_valid)
    max_diff = np.max(x_diff)
    
    print(f"\nMaximum X(TI) difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ Results match within numerical tolerance!")
    else:
        print("⚠️  Results diverge significantly")
        print("  This indicates the CPU/GPU divergence issue is still present")
        
elif cpu_success and not gpu_success:
    print("\n=== Summary ===")
    print("✓ CPU calculation works")
    print("✗ GPU calculation fails")
    print("\nThis is expected if the GPU code has compilation errors.")
    print("The GPU code needs to be fixed before comparison can proceed.")
else:
    print("\n=== Summary ===")
    print("Unexpected failure in CPU calculation")