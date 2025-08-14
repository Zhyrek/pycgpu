#!/usr/bin/env python
"""Test exact point conditions without creating a grid."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_exact_conditions():
    """Test that we can specify exact point conditions."""
    
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']
    
    print("Testing exact conditions specification...")
    
    # Method 1: Single condition (always works)
    print("\n1. Single condition:")
    conditions = {
        v.X('AL'): 0.3,
        v.X('CU'): 0.2,
        v.T: 900,
        v.P: 101325
    }
    result = equilibrium(dbf, comps, phases, conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"  Shape: {result.GM.shape}")
    print(f"  GM: {result.GM.values.item():.2f}")
    
    # Method 2: Multiple conditions using numpy arrays
    print("\n2. Multiple conditions with numpy arrays:")
    conditions = {
        v.X('AL'): np.array([0.1, 0.3, 0.5]),
        v.X('CU'): np.array([0.1, 0.2, 0.3]),
        v.T: np.array([600, 900, 1200]),
        v.P: 101325
    }
    try:
        result = equilibrium(dbf, comps, phases, conditions,
                            calc_opts={'pdens': 50}, gpu=True, verbose=False)
        print(f"  Shape: {result.GM.shape}")
        print(f"  Number of results: {result.GM.size}")
        for i in range(min(3, result.GM.size)):
            print(f"  GM[{i}]: {result.GM.values.flat[i]:.2f}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Method 3: Lists (creates a grid)
    print("\n3. Lists (creates grid):")
    conditions = {
        v.X('AL'): [0.1, 0.3, 0.5],
        v.X('CU'): [0.1, 0.2, 0.3],
        v.T: [600, 900, 1200],
        v.P: 101325
    }
    result = equilibrium(dbf, comps, phases, conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"  Shape: {result.GM.shape}")
    print(f"  Number of results (should be 3x3x3=27): {result.GM.size}")
    
    # Method 4: Keep one dimension constant
    print("\n4. Lists with one constant dimension:")
    conditions = {
        v.X('AL'): [0.1, 0.3, 0.5],
        v.X('CU'): [0.1, 0.2, 0.3],
        v.T: 900,  # Constant
        v.P: 101325
    }
    result = equilibrium(dbf, comps, phases, conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"  Shape: {result.GM.shape}")
    print(f"  Number of results (should be 3x3=9): {result.GM.size}")

if __name__ == "__main__":
    test_exact_conditions()