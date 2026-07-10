#!/usr/bin/env python
"""
Test AMD GPU fix for multiple conditions
"""
import os
os.environ['HIP_LAUNCH_BLOCKING'] = '1'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

db = Database("Al-Cu-Fe.tdb")

print("Testing AMD fix for bounds checking...")
print("=" * 60)

# Test with 2 conditions (the case that was failing)
print("\nTesting 2 conditions...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: [1000, 1100], v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    gm_values = result.GM.values.flatten()
    print(f"✓ SUCCESS with 2 conditions!")
    print(f"  GM values: {gm_values}")
except Exception as e:
    print(f"✗ FAILED with 2 conditions: {str(e)[:100]}")

# Test with 10 conditions
print("\nTesting 10 conditions...")
try:
    result = equilibrium(
        db,
        ['AL', 'CU', 'FE', 'VA'],
        'LIQUID',
        {v.T: np.linspace(900, 1200, 10), v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3},
        gpu=True
    )
    print(f"✓ SUCCESS with 10 conditions!")
    print(f"  First GM: {result.GM.values.flat[0]:.2f}, Last GM: {result.GM.values.flat[-1]:.2f}")
except Exception as e:
    print(f"✗ FAILED with 10 conditions: {str(e)[:100]}")

print("\n" + "=" * 60)
print("AMD fix test complete!")
print("\nThe fix moves bounds checking BEFORE pointer arithmetic,")
print("preventing AMD GPUs from faulting on invalid address calculations.")