#!/usr/bin/env python
"""Test single condition to verify gradient fix works."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os

# Clear GPU cache
if 'CUDA_CACHE_DISABLE' in os.environ:
    import cupy as cp
    cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test single condition
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing single condition after gradient fix...")
print(f"Conditions: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print("="*80)

try:
    # Run CPU calculation
    print("\nRunning CPU calculation...")
    cpu_result = equilibrium(db, components, phases, conditions, 
                            calc_opts={'pdens': 50}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    print(f"CPU GM: {cpu_gm:.6f}")
    
    # Run GPU calculation
    print("\nRunning GPU calculation...")
    gpu_result = equilibrium(db, components, phases, conditions, 
                            calc_opts={'pdens': 50}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    print(f"GPU GM: {gpu_gm:.6f}")
    
    # Compare results
    gm_diff = abs(cpu_gm - gpu_gm)
    print(f"\nGM difference: {gm_diff:.6f}")
    
    if gm_diff < 0.001:
        print("✓ PASS: CPU and GPU agree within tolerance!")
    else:
        print("✗ FAIL: CPU and GPU results differ significantly")
        
        # Show phase information
        print("\nPhase information:")
        cpu_phases = cpu_result.Phase.values[0][cpu_result.NP.values[0] > 0]
        gpu_phases = gpu_result.Phase.values[0][gpu_result.NP.values[0] > 0]
        
        print(f"CPU phases: {cpu_phases}")
        print(f"GPU phases: {gpu_phases}")
        
        # Show phase amounts
        cpu_np = cpu_result.NP.values[0][cpu_result.NP.values[0] > 0]
        gpu_np = gpu_result.NP.values[0][gpu_result.NP.values[0] > 0]
        
        print(f"\nCPU phase amounts: {cpu_np}")
        print(f"GPU phase amounts: {gpu_np}")

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)