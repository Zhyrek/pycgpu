#!/usr/bin/env python
"""Test just the specific failing conditions."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test conditions around the failing ones
# Failing: X(BI)=0.3, T=500K and X(BI)=0.2, T=600K
x_bi_values = [0.2, 0.3, 0.4]
temp_values = [500, 600]

print("Running CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=False, verbose=False)

print("Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=False)

print("\nResults:")
print("Condition | X(BI) | T(K) | CPU GM      | GPU GM      | Diff     | Status")
print("-" * 70)

for t_idx, temp in enumerate(temp_values):
    for x_idx, x_bi in enumerate(x_bi_values):
        cpu_gm = result_cpu.GM.values[0,0,t_idx,x_idx]
        gpu_gm = result_gpu.GM.values[0,0,t_idx,x_idx]
        diff = abs(cpu_gm - gpu_gm)
        status = "PASS" if diff < 1.0 else "FAIL"
        cond_idx = t_idx * len(x_bi_values) + x_idx
        print(f"{cond_idx:9d} | {x_bi:5.1f} | {temp:4d} | {cpu_gm:11.2f} | {gpu_gm:11.2f} | {diff:8.2f} | {status}")