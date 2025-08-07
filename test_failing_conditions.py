#!/usr/bin/env python
"""Test the two failing conditions in detail."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

# Test the two failing conditions
conditions = [
    {'X(BI)': 0.3, 'T': 500, 'P': 101325},
    {'X(BI)': 0.2, 'T': 600, 'P': 101325}
]

for cond in conditions:
    print(f"\n{'='*60}")
    print(f"Testing X(BI)={cond['X(BI)']: 0.1f}, T={cond['T']}K")
    print('='*60)
    
    # CPU calculation
    result_cpu = equilibrium(dbf, comps, phases, 
                            {v.X('BI'): cond['X(BI)'], v.T: cond['T'], v.P: cond['P']}, 
                            gpu=False, verbose=False)
    
    # GPU calculation with verbose output
    print("\nGPU calculation with verbose output:")
    result_gpu = equilibrium(dbf, comps, phases, 
                            {v.X('BI'): cond['X(BI)'], v.T: cond['T'], v.P: cond['P']}, 
                            gpu=True, verbose=True)
    
    print(f"\nCPU GM: {result_cpu.GM.values[0,0,0,0]:.6f}")
    print(f"GPU GM: {result_gpu.GM.values[0,0,0,0]:.6f}")
    print(f"Difference: {abs(result_cpu.GM.values[0,0,0,0] - result_gpu.GM.values[0,0,0,0]):.6f}")
    
    print(f"\nCPU MU(AU): {result_cpu.MU.values[0,0,0,0,0]:.6f}")
    print(f"GPU MU(AU): {result_gpu.MU.values[0,0,0,0,0]:.6f}")
    print(f"CPU MU(BI): {result_cpu.MU.values[0,0,0,0,1]:.6f}")
    print(f"GPU MU(BI): {result_gpu.MU.values[0,0,0,0,1]:.6f}")
    
    # Check phase amounts
    print("\nPhase amounts:")
    for phase_idx, phase in enumerate(phases):
        cpu_np = result_cpu.NP.values[0,0,0,0,phase_idx]
        gpu_np = result_gpu.NP.values[0,0,0,0,phase_idx]
        if cpu_np > 1e-10 or gpu_np > 1e-10:
            print(f"  {phase}: CPU={cpu_np:.6f}, GPU={gpu_np:.6f}")