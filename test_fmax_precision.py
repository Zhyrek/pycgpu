#!/usr/bin/env python
"""Test floating-point precision differences between fmax and max."""

import numpy as np

# Test values that might occur during consolidation
test_values = [
    0.179268138771215,   # Typical phase amount
    0.820731861228785,   # Typical phase amount
    1.0,                 # Sum of two phases
    0.999999999999999,   # Nearly 1.0
    1.000000000000001,   # Slightly over 1.0
]

print("Testing fmax vs max precision differences")
print("="*80)

for val in test_values:
    # Python max
    py_max = max(val, 1e-8)
    
    # Simulate C fmax behavior (numpy's maximum is equivalent)
    c_fmax = np.maximum(val, 1e-8)
    
    # Difference
    diff = c_fmax - py_max
    
    print(f"\nValue: {val:.15f}")
    print(f"  Python max: {py_max:.15f}")
    print(f"  C fmax:     {c_fmax:.15f}")
    print(f"  Difference: {diff:.15e}")

# Test the actual consolidation case
print("\n" + "="*80)
print("Simulating consolidation:")
phase1 = 0.179268138771215
phase2 = 0.820731861228785
total = phase1 + phase2

py_result = max(total, 1e-8)
c_result = np.maximum(total, 1e-8)

print(f"Phase 1: {phase1:.15f}")
print(f"Phase 2: {phase2:.15f}")
print(f"Total:   {total:.15f}")
print(f"Python max(total, 1e-8): {py_result:.15f}")
print(f"C fmax(total, 1e-8):     {c_result:.15f}")
print(f"Difference: {c_result - py_result:.15e}")

# Test if the issue is in the total calculation itself
print("\n" + "="*80)
print("Testing phase amount sum precision:")
# Using exact decimal representations
from decimal import Decimal, getcontext
getcontext().prec = 50

d_phase1 = Decimal('0.179268138771215')
d_phase2 = Decimal('0.820731861228785')
d_total = d_phase1 + d_phase2

print(f"Decimal sum: {d_total}")
print(f"Float sum:   {phase1 + phase2:.20f}")
print(f"Exact 1.0:   1.00000000000000000000")
print(f"Difference from 1.0: {(phase1 + phase2) - 1.0:.15e}")