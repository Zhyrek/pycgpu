#!/usr/bin/env python3
"""Trace GPU phase removal in detail"""
import os
import numpy as np

# Enable GPU debug output
os.environ['PYCALPHAD_DEBUG'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear any cached GPU modules
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== GPU Phase Removal Trace ===")
print("Looking for where second BCC_A2 phase is removed...")
print("\nRunning GPU calculation with debug output...")

# Run GPU calculation
try:
    gpu_result = equilibrium(db, comps, phases, conditions, verbose=True, gpu=True, calc_opts={'pdens': 50})
except Exception as e:
    print(f"GPU calculation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Analysis ===")
print("Key things to look for in the debug output above:")
print("1. Initial phases from starting point (should show 2 BCC_A2 phases)")
print("2. When phase amounts change during iterations")
print("3. Messages about 'Phase X marked for removal'")
print("4. The iteration where second BCC_A2 disappears")