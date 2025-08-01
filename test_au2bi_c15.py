#!/usr/bin/env python
"""
Test AU2BI_C15 + LIQUID phases
AU2BI_C15 has site ratios (2.0, 1.0) = 3.0 total, NO vacancy
"""

from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'AU2BI_C15']

# Single test condition
conditions = {
    'T': 400,
    'P': 101325,
    'X(BI)': 0.3  # Adjusted for AU2BI stoichiometry
}

print("Testing AU2BI_C15 + LIQUID phases...")
print(f"Conditions: {conditions}")
print("AU2BI_C15: site ratios (2.0, 1.0) = 3.0 total, NO vacancy")
print("LIQUID: site ratios (1.0,) = 1.0 total, NO vacancy")

# CPU calculation
print("\nCPU:")
result_cpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 1000})
cpu_gm = result_cpu.GM.values.flat[0]
print(f"  GM: {cpu_gm:.6f}")
for phase in np.unique(result_cpu.Phase.values):
    if phase != '':
        mask = result_cpu.Phase.values == phase
        amount = result_cpu.NP.values[mask][0]
        if amount > 1e-10:
            print(f"  {phase}: {amount:.6f}")

# GPU calculation
print("\nGPU:")
try:
    result_gpu = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 1000}, gpu=True)
    gpu_gm = result_gpu.GM.values.flat[0]
    print(f"  GM: {gpu_gm:.6f}")
    for phase in np.unique(result_gpu.Phase.values):
        if phase != '':
            mask = result_gpu.Phase.values == phase
            amount = result_gpu.NP.values[mask][0]
            if amount > 1e-10:
                print(f"  {phase}: {amount:.6f}")
    
    # Compare
    print(f"\nGM difference: {abs(cpu_gm - gpu_gm):.6f}")
    if abs(cpu_gm - gpu_gm) < 1.0:
        print("✓ Results match well!")
    else:
        print("✗ Large difference detected")
        
except Exception as e:
    print(f"GPU Error: {e}")
    import traceback
    traceback.print_exc()