#!/usr/bin/env python
"""Test GPU ALCU_ZETA phase compilation with a single condition"""

import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium
import numpy as np
from pycalphad.variables import N, T, P, X

# Create database
db = Database('Al-Cu-Fe.tdb')

# Single test condition
components = ['AL', 'CU', 'FE']
phases = ['ALCU_ZETA']

# Single condition for fast test
conditions = {
    N: 1.0,
    P: 101325,
    T: 1000,
    X('CU'): 0.5,
    X('FE'): 0.4
}

try:
    print("Testing ALCU_ZETA GPU compilation...")
    
    # Run equilibrium calculation with GPU
    result = equilibrium(db, components, phases, conditions, gpu=True, verbose=True)
    
    print("GPU compilation and calculation successful!")
    print(f"Result shape: {result.NP.shape}")
    print(f"Phase amounts: {result.NP.values}")
    
except Exception as e:
    print(f"GPU test failed: {e}")
    import traceback
    traceback.print_exc()