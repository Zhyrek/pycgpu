#!/usr/bin/env python
"""Test to verify that GPU threads are accessing the correct memory for their assigned conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_memory_access():
    """Test if threads are correctly accessing their assigned memory regions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    # Create conditions with very distinct compositions
    # Use compositions that would be easy to identify if mixed up
    test_conditions = [
        (0.10, 0.10, 600),   # Condition 0: Low Al, Low Cu
        (0.90, 0.05, 600),   # Condition 1: High Al, Low Cu  
        (0.05, 0.90, 600),   # Condition 2: Low Al, High Cu
        (0.50, 0.20, 900),   # Condition 3: Mid Al, Low Cu
        (0.20, 0.50, 900),   # Condition 4: Low Al, Mid Cu
        (0.30, 0.30, 1200),  # Condition 5: Equal Al/Cu
        (0.70, 0.10, 1200),  # Condition 6: High Al, Low Cu
        (0.10, 0.70, 1200),  # Condition 7: Low Al, High Cu
    ]
    
    print("=" * 80)
    print("TESTING THREAD MEMORY ACCESS PATTERNS")
    print("=" * 80)
    
    # Test 1: Run each condition individually
    print("\n1. Individual execution (reference):")
    individual_results = []
    for idx, (x_al, x_cu, temp) in enumerate(test_conditions):
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        result = equilibrium(dbf, comps, phases, conditions,
                           calc_opts={'pdens': 50},
                           gpu=True, verbose=False)
        gm = result.GM.values.item()
        individual_results.append(gm)
        print(f"  Condition {idx}: X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, T={temp}K -> GM={gm:.2f}")
    
    # Test 2: Run all conditions in a batch
    print("\n2. Batch execution (all at once):")
    x_al_vals = [c[0] for c in test_conditions]
    x_cu_vals = [c[1] for c in test_conditions]
    temp_vals = [c[2] for c in test_conditions]
    
    conditions_batch = {
        v.X('AL'): x_al_vals,
        v.X('CU'): x_cu_vals,
        v.T: temp_vals,
        v.P: 101325
    }
    
    result_batch = equilibrium(dbf, comps, phases, conditions_batch,
                             calc_opts={'pdens': 50},
                             gpu=True, verbose=False)
    batch_results = result_batch.GM.values.flatten()
    
    # Compare results
    print("\n3. Comparison:")
    print("-" * 60)
    print("Idx | X(AL) | X(CU) | T(K) | Individual GM | Batch GM | Diff | Status")
    print("-" * 60)
    
    all_pass = True
    for idx, (x_al, x_cu, temp) in enumerate(test_conditions):
        ind_gm = individual_results[idx]
        batch_gm = batch_results[idx]
        diff = abs(batch_gm - ind_gm)
        status = "✓" if diff < 1.0 else "✗"
        if diff >= 1.0:
            all_pass = False
        
        print(f"{idx:3d} | {x_al:5.2f} | {x_cu:5.2f} | {temp:4.0f} | {ind_gm:13.2f} | {batch_gm:8.2f} | {diff:4.2f} | {status}")
    
    print("-" * 60)
    
    # Test 3: Check for memory access patterns
    print("\n4. Memory Access Pattern Analysis:")
    
    # Look for patterns in failures
    failed_indices = []
    for idx in range(len(test_conditions)):
        if abs(batch_results[idx] - individual_results[idx]) >= 1.0:
            failed_indices.append(idx)
    
    if failed_indices:
        print(f"  Failed condition indices: {failed_indices}")
        
        # Check if failures follow a pattern
        if len(failed_indices) > 1:
            diffs = [failed_indices[i+1] - failed_indices[i] for i in range(len(failed_indices)-1)]
            if all(d == diffs[0] for d in diffs):
                print(f"  Pattern detected: Failures occur every {diffs[0]} conditions")
            
        # Check if batch results match wrong individual results
        print("\n  Checking for memory cross-contamination:")
        for fail_idx in failed_indices:
            batch_val = batch_results[fail_idx]
            # Check if this value matches any other condition's individual result
            for check_idx, ind_val in enumerate(individual_results):
                if check_idx != fail_idx and abs(batch_val - ind_val) < 1.0:
                    print(f"    Condition {fail_idx} batch result matches condition {check_idx} individual result!")
                    print(f"      This suggests thread {fail_idx} is reading memory from thread {check_idx}")
    else:
        print("  No failures detected - memory access appears correct")
    
    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL TESTS PASSED - Thread memory isolation is working correctly")
    else:
        print("✗ MEMORY ACCESS ISSUES DETECTED")
        print(f"  {len(failed_indices)} out of {len(test_conditions)} conditions failed")
        print("  This indicates threads are accessing incorrect memory regions")
    print("=" * 80)

if __name__ == "__main__":
    test_memory_access()