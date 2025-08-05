#!/usr/bin/env python
"""Test 5 phases with GPU to debug crash."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_5phases():
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    # Test with exactly 5 phases
    phases_5 = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID']
    
    print(f"Testing with 5 phases: {phases_5}")
    
    # Test with minimal conditions
    for num_conds in [1, 2, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        X_vals = np.linspace(0.1, 0.9, num_conds)
        conds = {v.X('BI'): X_vals, v.T: 400, v.P: 101325}
        
        try:
            result_gpu = equilibrium(dbf, comps, phases_5, conds, gpu=True, verbose=False)
            print(f"✓ Success with {num_conds} conditions")
        except Exception as e:
            print(f"✗ Failed with {num_conds} conditions: {type(e).__name__}")
            if num_conds > 1:
                # Found the breaking point
                print(f"\nBreaks between {last_good} and {num_conds} conditions")
                break
        
        last_good = num_conds

if __name__ == "__main__":
    test_5phases()