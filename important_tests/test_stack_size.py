#!/usr/bin/env python
"""Test to determine if the illegal memory access is due to stack overflow."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_stack_overflow():
    """Test if the issue is related to stack size by varying the number of conditions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("TESTING FOR STACK OVERFLOW WITH VARYING NUMBER OF CONDITIONS")
    print("=" * 80)
    
    # Test with increasing number of conditions
    for num_conditions in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f"\nTesting with {num_conditions} conditions...")
        
        # Create distinct conditions
        x_al_vals = []
        x_cu_vals = []
        temp_vals = []
        
        for i in range(num_conditions):
            x_al = 0.1 + i * 0.1
            x_cu = 0.1 + (num_conditions - i - 1) * 0.05
            # Ensure valid composition
            if x_al + x_cu < 0.95:
                x_al_vals.append(x_al)
                x_cu_vals.append(x_cu)
                temp_vals.append(600 + i * 100)
        
        if len(x_al_vals) == 0:
            print(f"  Skipping {num_conditions} conditions (invalid compositions)")
            continue
            
        conditions = {
            v.X('AL'): x_al_vals if len(x_al_vals) > 1 else x_al_vals[0],
            v.X('CU'): x_cu_vals if len(x_cu_vals) > 1 else x_cu_vals[0],
            v.T: temp_vals if len(temp_vals) > 1 else temp_vals[0],
            v.P: 101325
        }
        
        try:
            result = equilibrium(dbf, comps, phases, conditions,
                               calc_opts={'pdens': 50},
                               gpu=True, verbose=False)
            
            # Check if we got valid results
            gm_values = result.GM.values.flatten()
            num_valid = np.sum(~np.isnan(gm_values))
            
            print(f"  ✓ SUCCESS: {len(x_al_vals)} conditions executed, {num_valid} valid results")
            
        except Exception as e:
            error_msg = str(e)
            if "cudaErrorIllegalAddress" in error_msg:
                print(f"  ✗ FAILED: Illegal memory access with {len(x_al_vals)} conditions")
                print(f"    This suggests stack overflow starts at {len(x_al_vals)} concurrent threads")
                break
            elif "cudaErrorLaunchFailure" in error_msg:
                print(f"  ✗ FAILED: Kernel launch failure with {len(x_al_vals)} conditions")
                print(f"    This suggests severe memory issues")
                break
            else:
                print(f"  ✗ FAILED: {error_msg[:100]}...")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("If failure occurs at low thread counts (2-4), it's likely a stack overflow issue.")
    print("The SystemState struct is very large and allocated on each thread's stack.")
    print("=" * 80)

if __name__ == "__main__":
    test_stack_overflow()