#!/usr/bin/env python
"""Test starting_point MU values for identical conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']

# First test with 4 phases (working case)
print("Testing with 4 phases:")
phases_4 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']
result_4 = equilibrium(dbf, comps, phases_4, 
                      {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}, 
                      gpu=False, verbose=False)

print(f"CPU MU shape: {result_4.MU.shape}")
print(f"CPU MU[0]: {result_4.MU.values[0,0,0,0]}")
print(f"CPU MU[1]: {result_4.MU.values[0,0,0,1]}")
print(f"Are they equal? {np.array_equal(result_4.MU.values[0,0,0,0], result_4.MU.values[0,0,0,1])}")

# Now test with 6 phases (failing case)
print("\n\nTesting with 6 phases:")
phases_6 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'AU2BI_HEX', 'RHOMBOHEDRAL_A7', 'BCC_A2']
result_6 = equilibrium(dbf, comps, phases_6, 
                      {v.X('BI'): [0.3, 0.3], v.T: 600, v.P: 101325}, 
                      gpu=False, verbose=False)

print(f"CPU MU shape: {result_6.MU.shape}")
print(f"CPU MU[0]: {result_6.MU.values[0,0,0,0]}")
print(f"CPU MU[1]: {result_6.MU.values[0,0,0,1]}")
print(f"Are they equal? {np.array_equal(result_6.MU.values[0,0,0,0], result_6.MU.values[0,0,0,1])}")

# The key finding: starting_point returns the same MU for identical conditions
# This is expected behavior! The issue must be elsewhere.