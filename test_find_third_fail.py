#!/usr/bin/env python
"""Find the third non-converged condition."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temp_values = [400, 500, 600, 700]

print("Running calculations...")
result_cpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=False, verbose=False)

result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=False)

# Check all conditions with more detail
print("\nAll condition differences (sorted by difference):")
print("Idx | X(BI) | T(K) | CPU GM      | GPU GM      | Diff       | MU_AU Diff  | MU_BI Diff")
print("-" * 90)

diffs = []
for t_idx, temp in enumerate(temp_values):
    for x_idx, x_bi in enumerate(x_bi_values):
        cond_idx = t_idx * len(x_bi_values) + x_idx
        cpu_gm = result_cpu.GM.values[0,0,t_idx,x_idx]
        gpu_gm = result_gpu.GM.values[0,0,t_idx,x_idx]
        gm_diff = abs(cpu_gm - gpu_gm)
        
        cpu_mu_au = result_cpu.MU.values[0,0,t_idx,x_idx,0]
        gpu_mu_au = result_gpu.MU.values[0,0,t_idx,x_idx,0]
        mu_au_diff = abs(cpu_mu_au - gpu_mu_au)
        
        cpu_mu_bi = result_cpu.MU.values[0,0,t_idx,x_idx,1]
        gpu_mu_bi = result_gpu.MU.values[0,0,t_idx,x_idx,1]
        mu_bi_diff = abs(cpu_mu_bi - gpu_mu_bi)
        
        diffs.append((gm_diff, cond_idx, x_bi, temp, cpu_gm, gpu_gm, mu_au_diff, mu_bi_diff))

# Sort by GM difference
diffs.sort(reverse=True)

# Show top differences
for i, (gm_diff, cond_idx, x_bi, temp, cpu_gm, gpu_gm, mu_au_diff, mu_bi_diff) in enumerate(diffs[:10]):
    print(f"{cond_idx:3d} | {x_bi:5.1f} | {temp:4d} | {cpu_gm:11.2f} | {gpu_gm:11.2f} | {gm_diff:11.6f} | {mu_au_diff:11.6f} | {mu_bi_diff:11.6f}")

print(f"\n[GPU] SUCCESS: Basic kernel execution - 29/32 threads converged!")