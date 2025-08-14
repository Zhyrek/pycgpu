#!/usr/bin/env python
"""Test Al-Cu-Fe system with 8 phases - NO PDENS."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_conditions_multi():
    """Test multiple conditions in Al-Cu-Fe system WITHOUT pdens."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    
    # Using 8 phases as before
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("Testing Al-Cu-Fe system with 8 phases - NO PDENS")
    print("=" * 80)
    
    # Test conditions - comprehensive grid
    al_vals = np.arange(0.1, 1.0, 0.1)
    cu_vals = np.arange(0.1, 1.0, 0.1)
    
    conditions_list = []
    for x_al in al_vals:
        for x_cu in cu_vals:
            if x_al + x_cu < 1.0:  # Valid composition
                conditions_list.append((x_al, x_cu, 900))  # 900K
                
    # Also test some at different temperatures
    for x_al in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for x_cu in [0.1, 0.2, 0.3, 0.4]:
            if x_al + x_cu < 1.0:
                conditions_list.append((x_al, x_cu, 600))   # 600K
                conditions_list.append((x_al, x_cu, 1200))  # 1200K
    
    # Remove duplicates
    conditions_list = list(set(conditions_list))
    
    print(f"Testing {len(conditions_list)} conditions")
    print("-" * 80)
    
    passed = 0
    failed = 0
    error_count = 0
    failures = []
    
    for i, (x_al, x_cu, temp) in enumerate(conditions_list):
        x_fe = 1.0 - x_al - x_cu
        
        conditions = {
            v.X('AL'): x_al,
            v.X('CU'): x_cu,
            v.T: temp,
            v.P: 101325
        }
        
        try:
            # CPU calculation - NO PDENS
            cpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=False, verbose=False)
            cpu_gm = cpu_result.GM.values.item()
            
            # GPU calculation - NO PDENS
            gpu_result = equilibrium(dbf, comps, phases, conditions,
                                    gpu=True, verbose=False)
            gpu_gm = gpu_result.GM.values.item()
            
            diff = abs(gpu_gm - cpu_gm)
            
            if diff < 100:  # 100 J/mol tolerance
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"
                failures.append((x_al, x_cu, temp, diff))
                
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(conditions_list)}] Passed: {passed}, Failed: {failed}, Errors: {error_count}")
                
        except Exception as e:
            error_count += 1
            status = "ERROR"
            print(f"  Error at X(AL)={x_al:.2f}, X(CU)={x_cu:.2f}, T={temp}: {str(e)[:50]}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total conditions tested: {len(conditions_list)}")
    print(f"Passed: {passed} ({100*passed/len(conditions_list):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(conditions_list):.1f}%)")
    print(f"Errors: {error_count}")
    
    if failures:
        print("\nFailing conditions (diff > 100 J/mol):")
        print("-" * 80)
        print("X(AL) | X(CU) | X(FE) | T(K) | Diff (J/mol)")
        print("------|-------|-------|------|-------------")
        
        # Sort by difference
        failures.sort(key=lambda x: x[3], reverse=True)
        
        for x_al, x_cu, temp, diff in failures[:20]:  # Show top 20
            x_fe = 1.0 - x_al - x_cu
            print(f"{x_al:.2f}  | {x_cu:.2f}  | {x_fe:.2f}  | {temp:4d} | {diff:12.1f}")
        
        if len(failures) > 20:
            print(f"... and {len(failures)-20} more failures")
    
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Analyze by temperature
    for temp in [600, 900, 1200]:
        temp_conds = [(x_al, x_cu, t) for x_al, x_cu, t in conditions_list if t == temp]
        if temp_conds:
            temp_pass = sum(1 for x_al, x_cu, t in temp_conds 
                          if not any((x_al, x_cu, t, d) for x_al, x_cu, t, d in failures))
            print(f"T={temp}K: {temp_pass}/{len(temp_conds)} passed ({100*temp_pass/len(temp_conds):.1f}%)")
    
    # Analyze by composition ranges
    print("\nBy composition range:")
    ranges = [
        ("Low Al (0.1-0.3)", 0.1, 0.3),
        ("Med Al (0.4-0.6)", 0.4, 0.6),
        ("High Al (0.7-0.9)", 0.7, 0.9)
    ]
    
    for name, al_min, al_max in ranges:
        range_conds = [(x_al, x_cu, t) for x_al, x_cu, t in conditions_list 
                      if al_min <= x_al <= al_max]
        if range_conds:
            range_pass = sum(1 for x_al, x_cu, t in range_conds 
                           if not any((x_al, x_cu, t, d) for x_al, x_cu, t, d in failures))
            print(f"{name}: {range_pass}/{len(range_conds)} passed ({100*range_pass/len(range_conds):.1f}%)")
    
    return passed, failed, error_count

if __name__ == "__main__":
    test_conditions_multi()