#!/usr/bin/env python
"""Add debug output to see what each thread is reading."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Run smaller test to see debug output
x_bi_values = [0.1, 0.2, 0.3, 0.4]
temp_values = [400, 500, 600]

print("Running GPU calculation with debug output...")
# This should be 12 conditions (4 * 3), so threads 0-11
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=True)

# Check which ones would correspond to our failing conditions
# Original condition 10: X(BI)=0.3, T=500K would be at [1,2] = condition 6
# Original condition 17: X(BI)=0.2, T=600K would be at [2,1] = condition 9

print("\nChecking specific conditions:")
print(f"Condition 6 (X=0.3, T=500): GM={result_gpu.GM.values[0,0,1,2]:.2f}")
print(f"Condition 9 (X=0.2, T=600): GM={result_gpu.GM.values[0,0,2,1]:.2f}")