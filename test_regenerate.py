#!/usr/bin/env python
"""
Force regeneration of GPU kernel
"""

from pycalphad import Database, equilibrium
import numpy as np
import os

# Clear cache
os.system('rm -rf ~/.cupy/kernel_cache/*')
os.system('rm -f generated_equilibrium_kernel.cu')

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

print("Forcing kernel regeneration...")

# GPU calculation with verbose=True to save generated code
result_gpu = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 1000}, 
                        gpu=True, 
                        verbose=True)

print("\nChecking if energy assignment was fixed...")
os.system('grep -n "csst->energy = " generated_equilibrium_kernel.cu | head -5')