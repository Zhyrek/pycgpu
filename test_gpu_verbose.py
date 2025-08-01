#!/usr/bin/env python
"""
Test GPU with verbose output to check generated code
"""

from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -f ~/.cupy/kernel_cache/*')

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

print("Testing GPU with verbose output...")

# GPU calculation with verbose=True to save generated code
result_gpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 1000}, 
                        gpu=True, 
                        verbose=True)

print("\nGenerated kernel code should be saved to: generated_equilibrium_kernel.cu")
print("Check if the normalization_factor fix is in the generated code.")