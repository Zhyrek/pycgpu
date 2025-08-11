#!/usr/bin/env python
"""Test pycalphad's range syntax."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID']

print("Testing pycalphad range syntax:")
print("=" * 50)

# Test 1: Single range
conditions1 = {
    v.T: (600, 1200, 200),  # start, stop, step
    v.P: 101325,
    v.X('CU'): 0.3,
    v.X('FE'): 0.2
}

wks1 = Workspace(dbf, comps, phases, conditions1)
print("\nTest 1 - T range (600, 1200, 200):")
print(f"  Result: {wks1.conditions[v.T]}")
print(f"  Length: {len(wks1.conditions[v.T])}")

# Test 2: Composition range
conditions2 = {
    v.T: 800,
    v.P: 101325,
    v.X('CU'): (0.1, 0.5, 0.1),
    v.X('FE'): 0.2
}

wks2 = Workspace(dbf, comps, phases, conditions2)
print("\nTest 2 - X(CU) range (0.1, 0.5, 0.1):")
print(f"  Result: {wks2.conditions[v.X('CU')]}")
print(f"  Length: {len(wks2.conditions[v.X('CU')])}")

# Test 3: Multiple ranges
conditions3 = {
    v.T: (600, 1000, 200),
    v.P: 101325,
    v.X('CU'): (0.1, 0.4, 0.1),
    v.X('FE'): (0.1, 0.3, 0.1)
}

wks3 = Workspace(dbf, comps, phases, conditions3)
print("\nTest 3 - Multiple ranges:")
print(f"  T range (600, 1000, 200): {wks3.conditions[v.T]}")
print(f"  X(CU) range (0.1, 0.4, 0.1): {wks3.conditions[v.X('CU')]}")
print(f"  X(FE) range (0.1, 0.3, 0.1): {wks3.conditions[v.X('FE')]}")

# Calculate total conditions
print(f"\nTotal conditions in Test 3:")
total = 1
for key, val in wks3.conditions.items():
    if hasattr(val, '__len__') and key != v.N and key != v.P:
        print(f"  {key}: {len(val)} values")
        total *= len(val) if len(val) > 1 else 1
print(f"  Expected product: {total}")