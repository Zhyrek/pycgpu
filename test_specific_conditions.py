#!/usr/bin/env python
"""Test the specific failing conditions to understand what's special about them."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = filter_phases(dbf, comps)

# Test the specific failing conditions
test_conditions = [
    (10, 0.3, 500),  # Thread 10
    (17, 0.2, 600),  # Thread 17
    # Also test neighbors for comparison
    (9,  0.2, 500),  # Before 10
    (11, 0.4, 500),  # After 10
    (16, 0.1, 600),  # Before 17
    (18, 0.3, 600),  # After 17
]

print("Testing specific conditions and their neighbors...")
print("="*60)

for idx, x_bi, temp in test_conditions:
    cond = {v.X('BI'): x_bi, v.T: temp, v.P: 101325}
    
    # Run both CPU and GPU
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    
    # Extract results
    gm_cpu = float(result_cpu.GM.values)
    gm_gpu = float(result_gpu.GM.values)
    phases_cpu = [p for p in result_cpu.Phase.values.flatten() if p and p != '']
    phases_gpu = [p for p in result_gpu.Phase.values.flatten() if p and p != '']
    np_cpu = result_cpu.NP.values[~np.isnan(result_cpu.NP.values)]
    np_gpu = result_gpu.NP.values[~np.isnan(result_gpu.NP.values)]
    
    diff = abs(gm_cpu - gm_gpu)
    status = "PASS" if diff < 1e-3 else "FAIL"
    
    print(f"\nIndex {idx}: X(BI)={x_bi}, T={temp}K - {status}")
    print(f"  CPU: GM={gm_cpu:.2f}, Phases={phases_cpu}, NP={np_cpu}")
    print(f"  GPU: GM={gm_gpu:.2f}, Phases={phases_gpu}, NP={np_gpu}")
    print(f"  Diff: {diff:.6f}")
    
    # Check if phase assemblage changes
    if idx in [10, 17] and status == "FAIL":
        print(f"  Note: This is a known failing condition (thread % 7 = {idx % 7})")