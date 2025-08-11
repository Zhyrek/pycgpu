#!/usr/bin/env python
"""Debug constraint coefficients."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Test Au-Bi binary
print("=" * 60)
print("TESTING Au-Bi CONSTRAINT SETUP (BINARY)")
print("=" * 60)
dbf = Database('important_tests/AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('BI'): 0.3
}

print(f"Components: {comps}")
non_va = [c for c in comps if c != 'VA']
print(f"Non-VA components: {non_va}, count: {len(non_va)}")
print(f"Condition: X(BI) = 0.3")

# The constraint matrix for Au-Bi should be:
# Coefficients: [1.0, 0.0, 0.0] for component BI (index 1)
# RHS: 0.3
print("\nExpected for binary:")
print("  Coefficients: [0.0, 1.0, 0.0] (1.0 for BI index)")
print("  RHS: 0.3")

# Test Al-Cu-Fe ternary
print("\n" + "=" * 60)
print("TESTING Al-Cu-Fe CONSTRAINT SETUP (TERNARY)")
print("=" * 60)
dbf2 = Database('Al-Cu-Fe.tdb')
comps2 = ['AL', 'CU', 'FE', 'VA']
phases2 = ['LIQUID', 'FCC_A1']
conditions2 = {
    v.T: 800,
    v.P: 101325,
    v.X('CU'): 0.1,
    v.X('FE'): 0.1
}

print(f"Components: {comps2}")
non_va2 = [c for c in comps2 if c != 'VA']
print(f"Non-VA components: {non_va2}, count: {len(non_va2)}")
print(f"Conditions: X(CU) = 0.1, X(FE) = 0.1")

# The constraint matrix for Al-Cu-Fe should be:
# For X(CU) = 0.1: Coefficients: [-0.1, 0.9, -0.1, 0.0] (0.9 for CU, -0.1 for others)
# For X(FE) = 0.1: Coefficients: [-0.1, -0.1, 0.9, 0.0] (0.9 for FE, -0.1 for others)
# RHS: [0.0, 0.0]
print("\nExpected for ternary:")
print("  Constraint 1 (X_CU=0.1): Coefficients=[-0.1, 0.9, -0.1, 0.0], RHS=0.0")
print("  Constraint 2 (X_FE=0.1): Coefficients=[-0.1, -0.1, 0.9, 0.0], RHS=0.0")