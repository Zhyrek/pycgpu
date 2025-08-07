#!/usr/bin/env python
"""Test failing conditions in the same multi-condition call as the original test."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test with the same multi-condition setup as the comprehensive test
x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temp_values = [400, 500, 600, 700]

# Run same conditions
print("Running CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=False, verbose=False)

print("Running GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=True)

# Check the specific failing conditions
# X(BI)=0.3, T=500K should be at indices [1,2] (T=500 is index 1, X_BI=0.3 is index 2)
# X(BI)=0.2, T=600K should be at indices [2,1] (T=600 is index 2, X_BI=0.2 is index 1)

print("\nChecking X(BI)=0.3, T=500K:")
cpu_gm_1 = result_cpu.GM.values[0,0,1,2]
gpu_gm_1 = result_gpu.GM.values[0,0,1,2]
print(f"CPU GM: {cpu_gm_1:.6f}")
print(f"GPU GM: {gpu_gm_1:.6f}")
print(f"Difference: {abs(cpu_gm_1 - gpu_gm_1):.6f}")

print("\nChecking X(BI)=0.2, T=600K:")
cpu_gm_2 = result_cpu.GM.values[0,0,2,1]
gpu_gm_2 = result_gpu.GM.values[0,0,2,1]
print(f"CPU GM: {cpu_gm_2:.6f}")
print(f"GPU GM: {gpu_gm_2:.6f}")
print(f"Difference: {abs(cpu_gm_2 - gpu_gm_2):.6f}")

# Check all conditions
print("\nAll GM differences:")
for t_idx, temp in enumerate(temp_values):
    for x_idx, x_bi in enumerate(x_bi_values):
        cpu_val = result_cpu.GM.values[0,0,t_idx,x_idx]
        gpu_val = result_gpu.GM.values[0,0,t_idx,x_idx]
        diff = abs(cpu_val - gpu_val)
        status = "FAIL" if diff > 1.0 else "PASS"
        print(f"X(BI)={x_bi:.1f}, T={temp}K: CPU={cpu_val:.2f}, GPU={gpu_val:.2f}, Diff={diff:.2f} {status}")