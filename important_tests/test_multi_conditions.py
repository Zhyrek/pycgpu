#!/usr/bin/env python
"""
Test multiple conditions to reproduce AMD GPU crash
"""
import os
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
os.environ['AMD_LOG_LEVEL'] = '2'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

db = Database("Al-Cu-Fe.tdb")

print("Testing with different numbers of conditions...")

# Test 1: Single condition (should work)
print("\n1. Testing 1 condition...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: 1000, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    print("   SUCCESS with 1 condition")
except Exception as e:
    print(f"   FAILED with 1 condition: {e}")

# Test 2: Two conditions
print("\n2. Testing 2 conditions...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: [1000, 1100], v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    print("   SUCCESS with 2 conditions")
except Exception as e:
    print(f"   FAILED with 2 conditions: {e}")

# Test 3: 256 conditions (exactly one block)
print("\n3. Testing 256 conditions (one block)...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: np.linspace(900, 1200, 256), v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    print("   SUCCESS with 256 conditions")
except Exception as e:
    print(f"   FAILED with 256 conditions: {e}")

# Test 4: 257 conditions (two blocks)
print("\n4. Testing 257 conditions (two blocks)...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: np.linspace(900, 1200, 257), v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    print("   SUCCESS with 257 conditions")
except Exception as e:
    print(f"   FAILED with 257 conditions: {e}")

print("\nDone testing.")