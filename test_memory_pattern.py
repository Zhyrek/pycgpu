#!/usr/bin/env python
"""Test to identify memory access patterns with 6 phases."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Test with 6 phases
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
print(f"Testing with {len(phases)} phases: {phases}")

# Test different condition patterns
test_cases = [
    ("Single condition", {v.X('BI'): 0.3, v.T: 600, v.P: 101325}),
    ("Two identical conditions", {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}),
    ("Two different X", {v.X('BI'): [0.3, 0.4], v.T: 600, v.P: 101325}),
    ("Two different T", {v.X('BI'): 0.3, v.T: [600, 700], v.P: 101325}),
    ("Three conditions", {v.X('BI'): [0.3, 0.3, 0.3], v.T: 600, v.P: 101325}),
]

for name, conditions in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    
    try:
        # CPU calculation
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        cpu_gm = result_cpu.GM.values.flatten()
        
        # GPU calculation
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
        gpu_gm = result_gpu.GM.values.flatten()
        
        # Compare
        diff = np.abs(cpu_gm - gpu_gm)
        max_diff = np.max(diff)
        
        print(f"  CPU GM: {cpu_gm}")
        print(f"  GPU GM: {gpu_gm}")
        print(f"  Max difference: {max_diff:.6f} J/mol")
        
        if max_diff < 1.0:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
            
            # Check if it's always the same wrong value
            if len(gpu_gm) > 1 and np.all(np.isclose(gpu_gm[1:], gpu_gm[0])):
                print(f"  WARNING: All GPU values are the same: {gpu_gm[0]}")
            
            # Check phase results
            cpu_phases = result_cpu.Phase.values
            gpu_phases = result_gpu.Phase.values
            print(f"  CPU phases shape: {cpu_phases.shape}")
            print(f"  GPU phases shape: {gpu_phases.shape}")
            
    except Exception as e:
        print(f"  ✗ ERROR: {type(e).__name__}: {e}")