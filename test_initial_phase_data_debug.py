#!/usr/bin/env python
"""Debug initial phase data for failing conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Just run a small subset including the failing conditions
x_bi_values = [0.1, 0.2, 0.3, 0.4]  # Include X(BI)=0.3
temp_values = [400, 500, 600]  # Include T=500 and T=600

print("Running GPU calculation with verbose output...")
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=True)

# Condition indices in this smaller test:
# X(BI)=0.3, T=500K should be at [1,2] (T=500 is index 1, X_BI=0.3 is index 2)
# X(BI)=0.2, T=600K should be at [2,1] (T=600 is index 2, X_BI=0.2 is index 1)
# In flat array: condition 6 and condition 9

print("\nChecking results:")
for t_idx, temp in enumerate(temp_values):
    for x_idx, x_bi in enumerate(x_bi_values):
        gm = result_gpu.GM.values[0,0,t_idx,x_idx]
        mu_au = result_gpu.MU.values[0,0,t_idx,x_idx,0]
        mu_bi = result_gpu.MU.values[0,0,t_idx,x_idx,1]
        cond_idx = t_idx * len(x_bi_values) + x_idx
        print(f"Condition {cond_idx}: X(BI)={x_bi:.1f}, T={temp}K: GM={gm:.2f}, MU(AU)={mu_au:.2f}, MU(BI)={mu_bi:.2f}")