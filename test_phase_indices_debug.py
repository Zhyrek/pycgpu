#!/usr/bin/env python
"""Debug phase indices mapping issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v

def test_phase_indices():
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # Test with different numbers of phases
    all_phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    for num_phases in [4, 5, 6]:
        phases = all_phases[:num_phases]
        print(f"\nTesting with {num_phases} phases: {phases}")
        
        # Single condition first
        conds = {v.X('BI'): 0.1, v.T: 400, v.P: 101325}
        try:
            result = equilibrium(dbf, comps, phases, conds, gpu=True, verbose=False)
            print(f"  ✓ Single condition works")
        except Exception as e:
            print(f"  ✗ Single condition failed: {type(e).__name__}")
            
        # Test with 14 conditions (last working)
        X_vals = np.linspace(0.1, 0.9, 14)
        conds = {v.X('BI'): X_vals, v.T: 400, v.P: 101325}
        try:
            result = equilibrium(dbf, comps, phases, conds, gpu=True, verbose=False)
            print(f"  ✓ 14 conditions work")
        except Exception as e:
            print(f"  ✗ 14 conditions failed: {type(e).__name__}")
            
        # Test with 15 conditions (first failing)
        X_vals = np.linspace(0.1, 0.9, 15)
        conds = {v.X('BI'): X_vals, v.T: 400, v.P: 101325}
        try:
            result = equilibrium(dbf, comps, phases, conds, gpu=True, verbose=False)
            print(f"  ✓ 15 conditions work")
        except Exception as e:
            print(f"  ✗ 15 conditions failed: {type(e).__name__}")

if __name__ == "__main__":
    test_phase_indices()