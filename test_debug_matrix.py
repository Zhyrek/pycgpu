#!/usr/bin/env python
"""
Debug equilibrium matrix for FCC normalization issue
"""

from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache and enable debug
os.system('rm -f ~/.cupy/kernel_cache/*')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VERBOSE_DEBUG'] = '1'

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single test condition
conditions = {
    'T': 400,
    'P': 101325,
    'X(BI)': 0.1
}

print("Testing GPU equilibrium matrix with debug output...")

# GPU calculation with verbose output
result_gpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 1000}, 
                        gpu=True,
                        verbose=False)  # Set to False to reduce output clutter

# Extract key debug info from output
gpu_gm = result_gpu.GM.values.flat[0]
print(f"\nGPU GM: {gpu_gm:.6f}")
for phase in np.unique(result_gpu.Phase.values):
    if phase != '':
        mask = result_gpu.Phase.values == phase
        amount = result_gpu.NP.values[mask][0]
        if amount > 1e-10:
            print(f"GPU {phase}: {amount:.6f}")