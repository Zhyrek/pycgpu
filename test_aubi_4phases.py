#!/usr/bin/env python
"""Test Au-Bi system with exactly 4 phases."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 4 phases
phases_4 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']
print(f"Testing with 4 phases: {phases_4}")

# Test conditions
conditions = {
    v.X('BI'): (0.1, 0.9, 0.2),  # 5 compositions
    v.T: (400, 800, 200),         # 3 temperatures  
    v.P: 101325
}

try:
    # CPU calculation
    print("\nRunning CPU calculation...")
    result_cpu = equilibrium(dbf, comps, phases_4, conditions, gpu=False, verbose=False)
    print(f"CPU calculation completed")
    
    # GPU calculation  
    print("\nRunning GPU calculation...")
    result_gpu = equilibrium(dbf, comps, phases_4, conditions, gpu=True, verbose=False)
    print(f"GPU calculation completed")
    
    # Compare results
    cpu_gm = result_cpu.GM.values.flatten()
    gpu_gm = result_gpu.GM.values.flatten()
    
    max_diff = np.max(np.abs(cpu_gm - gpu_gm))
    avg_diff = np.mean(np.abs(cpu_gm - gpu_gm))
    
    print(f"\nResults comparison:")
    print(f"  Maximum GM difference: {max_diff:.6f} J/mol")
    print(f"  Average GM difference: {avg_diff:.6f} J/mol")
    
    if max_diff < 1.0:
        print(f"\n✓ TEST PASSED with 4 phases")
    else:
        print(f"\n✗ TEST FAILED with 4 phases - diff = {max_diff:.1f} J/mol")
        
except Exception as e:
    print(f"\n✗ ERROR with 4 phases: {type(e).__name__}: {e}")

# Now test with 5 phases
phases_5 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7', 'BCC_A2']
print(f"\n{'='*60}")
print(f"Testing with 5 phases: {phases_5}")

try:
    # GPU calculation with 5 phases
    print("\nRunning GPU calculation with 5 phases...")
    result_gpu_5 = equilibrium(dbf, comps, phases_5, conditions, gpu=True, verbose=False)
    print(f"✓ GPU calculation completed successfully with 5 phases!")
    
except Exception as e:
    print(f"✗ ERROR with 5 phases: {type(e).__name__}: {e}")