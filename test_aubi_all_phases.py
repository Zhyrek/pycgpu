#!/usr/bin/env python
"""Test full Au-Bi system with all phases."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# Get all phases
all_phases = filter_phases(dbf, comps)
print(f"Testing full Au-Bi system with {len(all_phases)} phases:")
for phase in all_phases:
    print(f"  - {phase}")

# Test conditions - grid of compositions and temperatures
conditions = {
    v.X('BI'): (0.1, 0.9, 0.2),  # 0.1, 0.3, 0.5, 0.7, 0.9
    v.T: (400, 800, 200),         # 400, 600, 800
    v.P: 101325
}

expected_conditions = 5 * 3  # 5 compositions × 3 temperatures = 15

print(f"\nTesting {expected_conditions} conditions")

try:
    # CPU calculation
    print("\nRunning CPU calculation...")
    result_cpu = equilibrium(dbf, comps, all_phases, conditions, gpu=False, verbose=False)
    print(f"CPU calculation completed")
    
    # GPU calculation  
    print("\nRunning GPU calculation...")
    result_gpu = equilibrium(dbf, comps, all_phases, conditions, gpu=True, verbose=False)
    print(f"GPU calculation completed")
    
    # Compare results
    cpu_gm = result_cpu.GM.values.flatten()
    gpu_gm = result_gpu.GM.values.flatten()
    
    max_diff = np.max(np.abs(cpu_gm - gpu_gm))
    avg_diff = np.mean(np.abs(cpu_gm - gpu_gm))
    
    print(f"\nResults comparison:")
    print(f"  Maximum GM difference: {max_diff:.6f} J/mol")
    print(f"  Average GM difference: {avg_diff:.6f} J/mol")
    
    # Check phase distributions
    cpu_phases = result_cpu.Phase.values
    gpu_phases = result_gpu.Phase.values
    
    # Count unique phases in results
    cpu_unique = set(p for p in cpu_phases.flatten() if p and p != '' and p != '_FAKE_')
    gpu_unique = set(p for p in gpu_phases.flatten() if p and p != '' and p != '_FAKE_')
    
    print(f"\nPhases found:")
    print(f"  CPU: {len(cpu_unique)} unique phases - {cpu_unique}")
    print(f"  GPU: {len(gpu_unique)} unique phases - {gpu_unique}")
    
    if max_diff < 1.0:  # 1 J/mol tolerance
        print(f"\n✓ TEST PASSED - GPU matches CPU within tolerance")
    else:
        print(f"\n✗ TEST FAILED - GPU differs from CPU by {max_diff:.1f} J/mol")
        
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()