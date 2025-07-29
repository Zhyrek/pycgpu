#!/usr/bin/env python
"""Test all cases from gpu_cpu_comparison_results.txt after gradient fix."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test cases from the original comparison
test_cases = [
    (0.1, 500),    # In miscibility gap
    (0.2, 600),    
    (0.3, 700),    
    (0.4, 800),    
    (0.5, 900),    
    (0.6, 1000),   
    (0.7, 1100),   
    (0.8, 1200),   
    (0.9, 1300),   # Single phase region
]

print("Testing all cases after gradient ordering fix...")
print("="*80)
print(f"{'X(TI)':<10} {'T(K)':<10} {'CPU_GM':<20} {'GPU_GM':<20} {'Difference':<15} {'Status'}")
print("-"*80)

all_passed = True
max_diff = 0.0

for x_ti, temp in test_cases:
    conditions = {
        v.T: temp,
        v.P: 101325,
        v.X('TI'): x_ti,
        v.N: 1
    }
    
    # CPU calculation
    cpu_result = equilibrium(db, components, phases, conditions, 
                            calc_opts={'pdens': 50}, verbose=False)
    cpu_gm = float(cpu_result.GM.values)
    
    # GPU calculation
    gpu_result = equilibrium(db, components, phases, conditions, 
                            calc_opts={'pdens': 50}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values)
    
    diff = abs(cpu_gm - gpu_gm)
    max_diff = max(max_diff, diff)
    
    status = "✓ PASS" if diff < 0.001 else "✗ FAIL"
    if diff >= 0.001:
        all_passed = False
        
    print(f"{x_ti:<10.1f} {temp:<10} {cpu_gm:<20.6f} {gpu_gm:<20.6f} {diff:<15.6f} {status}")

print("-"*80)
print(f"\nMaximum difference: {max_diff:.6f}")
print(f"Overall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
print("="*80)