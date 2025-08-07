#!/usr/bin/env python
"""Check convergence details for all conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temp_values = [400, 500, 600, 700]

print("Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=False)

# Extract converged field if available
if hasattr(result_gpu, 'converged'):
    print("\nConvergence status by condition:")
    for t_idx, temp in enumerate(temp_values):
        for x_idx, x_bi in enumerate(x_bi_values):
            cond_idx = t_idx * len(x_bi_values) + x_idx
            # Check if converged field exists and access it
            try:
                converged = result_gpu.converged.values[0,0,t_idx,x_idx]
            except:
                converged = None
            gm = result_gpu.GM.values[0,0,t_idx,x_idx]
            print(f"Condition {cond_idx:2d}: X(BI)={x_bi:.1f}, T={temp}K, GM={gm:12.2f}, Converged={converged}")
else:
    print("No convergence information available")