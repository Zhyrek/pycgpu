#!/usr/bin/env python
"""Debug stride calculations in GPU kernel."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def run_test():
    """Test stride calculations with verbose output."""
    
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'BCC_B2', 'L12', 'ALCU_THETA', 'AL13FE4', 'AL5FE2']
    
    print("=" * 80)
    print("STRIDE DEBUG TEST")
    print("=" * 80)
    
    # Test with multiple conditions
    conditions = {
        v.X('AL'): [0.1, 0.3],  # Two conditions
        v.X('CU'): [0.1, 0.3],
        v.T: 600,
        v.P: 101325
    }
    
    print("\nRunning equilibrium calculation with 2 conditions...")
    print("Conditions:")
    print("  1: X(AL)=0.1, X(CU)=0.1, X(FE)=0.8, T=600K")
    print("  2: X(AL)=0.3, X(CU)=0.3, X(FE)=0.4, T=600K")
    
    # Run with verbose to see stride information
    result = equilibrium(dbf, comps, phases, conditions,
                        calc_opts={'pdens': 50},
                        gpu=True, verbose=True)
    
    print("\nResults:")
    gm_values = result.GM.values.flatten()
    for i, gm in enumerate(gm_values):
        print(f"  Condition {i+1}: GM = {gm:.6f}")

if __name__ == "__main__":
    run_test()