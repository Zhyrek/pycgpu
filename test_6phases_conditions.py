#!/usr/bin/env python
"""Test Au-Bi system with 6 phases under varying conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Full Au-Bi system with all phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
print(f"Testing with {len(phases)} phases: {phases}")

# Test with varying number of conditions
for num_conditions in [1, 2, 3, 4, 5, 10]:
    print(f"\n{'='*60}")
    print(f"Testing with {num_conditions} conditions")
    
    if num_conditions == 1:
        conditions = {
            v.X('BI'): 0.3,
            v.T: 600,
            v.P: 101325
        }
    else:
        # Multiple conditions
        x_vals = np.linspace(0.1, 0.9, num_conditions)
        conditions = {
            v.X('BI'): x_vals,
            v.T: 600,
            v.P: 101325
        }
    
    try:
        # CPU calculation
        print(f"  Running CPU calculation...")
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = result_cpu.GM.values.flatten()
        print(f"  ✓ CPU calculation completed")
        
        # GPU calculation
        print(f"  Running GPU calculation...")
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = result_gpu.GM.values.flatten()
        print(f"  ✓ GPU calculation completed")
        
        # Compare results
        max_diff = np.max(np.abs(cpu_gm - gpu_gm))
        avg_diff = np.mean(np.abs(cpu_gm - gpu_gm))
        
        print(f"  Maximum GM difference: {max_diff:.6f} J/mol")
        print(f"  Average GM difference: {avg_diff:.6f} J/mol")
        
        if max_diff < 1.0:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED - difference too large")
            
    except Exception as e:
        print(f"  ✗ ERROR: {type(e).__name__}: {e}")
        if "cudaErrorIllegalAddress" in str(e):
            print(f"  GPU memory access error detected")
            break

print(f"\n{'='*60}")
print("Summary: 6 phases test complete")