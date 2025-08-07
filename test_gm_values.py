#!/usr/bin/env python
"""Check if the wrong GM values match other conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temp_values = [400, 500, 600, 700]

# Run CPU calculation to get correct values
result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=False, verbose=False)

print("Looking for GM values close to the wrong GPU values:")
print("Target 1: -26853.83 (wrong value for condition 10)")
print("Target 2: -34593.93 (wrong value for condition 17)")
print()

for t_idx, temp in enumerate(temp_values):
    for x_idx, x_bi in enumerate(x_bi_values):
        gm = result_cpu.GM.values[0,0,t_idx,x_idx]
        cond_idx = t_idx * len(x_bi_values) + x_idx
        
        # Check if this GM is close to either wrong value
        if abs(gm - (-26853.83)) < 1.0:
            print(f"MATCH 1: Condition {cond_idx}: X(BI)={x_bi:.1f}, T={temp}K has GM={gm:.2f}")
        if abs(gm - (-34593.93)) < 1.0:
            print(f"MATCH 2: Condition {cond_idx}: X(BI)={x_bi:.1f}, T={temp}K has GM={gm:.2f}")