#!/usr/bin/env python
"""Analyze memory access patterns for different thread configurations."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases_4 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'RHOMBOHEDRAL_A7']
phases_6 = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test various thread counts
test_configs = [
    # (n_temps, n_comps, phases, description)
    (2, 5, phases_4, "10 conditions, 4 phases"),
    (3, 4, phases_4, "12 conditions, 4 phases"),
    (4, 4, phases_4, "16 conditions, 4 phases"),
    (4, 5, phases_4, "20 conditions, 4 phases"),
    (4, 6, phases_4, "24 conditions, 4 phases"),
    (4, 7, phases_4, "28 conditions, 4 phases"),
    (4, 8, phases_4, "32 conditions, 4 phases"),
    (2, 5, phases_6, "10 conditions, 6 phases"),
    (3, 4, phases_6, "12 conditions, 6 phases"),
    (4, 4, phases_6, "16 conditions, 6 phases"),
    (4, 5, phases_6, "20 conditions, 6 phases"),
    (4, 6, phases_6, "24 conditions, 6 phases"),
    (4, 7, phases_6, "28 conditions, 6 phases"),
    (4, 8, phases_6, "32 conditions, 6 phases"),
]

print("Testing various configurations...")
print("Config | Total | Phases | Description                | Result")
print("-" * 70)

for n_temps, n_comps, phases, desc in test_configs:
    temps = np.linspace(400, 700, n_temps)
    x_bi = np.linspace(0.1, 0.8, n_comps)
    total = n_temps * n_comps
    
    try:
        result = equilibrium(dbf, comps, phases, 
                           {v.X('BI'): x_bi, v.T: temps, v.P: 101325}, 
                           gpu=True, verbose=False)
        
        # Check for any large differences vs CPU
        result_cpu = equilibrium(dbf, comps, phases, 
                               {v.X('BI'): x_bi, v.T: temps, v.P: 101325}, 
                               gpu=False, verbose=False)
        
        max_diff = np.max(np.abs(result.GM.values - result_cpu.GM.values))
        
        if max_diff > 1.0:
            print(f"{total:6d} | {len(phases):6d} | {desc:26s} | FAIL (max diff: {max_diff:.2f})")
        else:
            print(f"{total:6d} | {len(phases):6d} | {desc:26s} | PASS")
            
    except Exception as e:
        print(f"{total:6d} | {len(phases):6d} | {desc:26s} | ERROR: {str(e)[:30]}...")