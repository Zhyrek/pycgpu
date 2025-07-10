#!/usr/bin/env python3
"""Simple final comparison of CPU vs GPU results"""
import os
os.environ['PYCALPHAD_DEBUG'] = '0'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear cached GPU modules
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Run CPU calculation
print("Running CPU calculation...")
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])

# Run GPU calculation
print("Running GPU calculation...")
gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])

print("\n=== Results ===")
print(f"CPU GM: {cpu_gm:.1f} J/mol")
print(f"GPU GM: {gpu_gm:.1f} J/mol")
print(f"Difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")

# Get the final compositions
cpu_y = cpu_result.Y.sel(vertex=0).values.squeeze()
gpu_y = gpu_result.Y.sel(vertex=0).values.squeeze()

print(f"\nCPU Y: {cpu_y}")
print(f"GPU Y: {gpu_y}")

if len(cpu_y) >= 2 and len(gpu_y) >= 2:
    print(f"\nSite fractions:")
    print(f"CPU: Y(NB)={cpu_y[0]:.4f}, Y(TI)={cpu_y[1]:.4f}")
    print(f"GPU: Y(NB)={gpu_y[0]:.4f}, Y(TI)={gpu_y[1]:.4f}")
    print(f"Difference: Y(NB)={abs(cpu_y[0]-gpu_y[0]):.6f}, Y(TI)={abs(cpu_y[1]-gpu_y[1]):.6f}")