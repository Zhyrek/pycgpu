#!/usr/bin/env python3
"""Check final compositions from CPU and GPU"""
import os
os.environ['PYCALPHAD_DEBUG'] = '0'  # Turn off debug

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

print("=== Final Composition Check ===")

# Run CPU calculation
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
cpu_phase = cpu_result.Phase.sel(vertex=0).values[0]

print("CPU Results:")
print(f"  GM: {cpu_gm:.1f} J/mol")
print(f"  Phase: {cpu_phase}")

# Get site fractions for the active phase
cpu_y = cpu_result.Y.sel(vertex=0).values
print(f"  Y(BCC_A2,0,NB) = {cpu_y[0]:.6f}")
print(f"  Y(BCC_A2,0,TI) = {cpu_y[1]:.6f}")

# Get mole fractions
cpu_x = cpu_result.X.sel(vertex=0).values  
print(f"  X(NB) = {cpu_x[0]:.6f}")
print(f"  X(TI) = {cpu_x[1]:.6f}")

# Run GPU calculation
gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])
gpu_phase = gpu_result.Phase.sel(vertex=0).values[0]

print("\nGPU Results:")
print(f"  GM: {gpu_gm:.1f} J/mol")
print(f"  Phase: {gpu_phase}")

# Get site fractions for the active phase
gpu_y = gpu_result.Y.sel(vertex=0).values
print(f"  Y(BCC_A2,0,NB) = {gpu_y[0]:.6f}")
print(f"  Y(BCC_A2,0,TI) = {gpu_y[1]:.6f}")

# Get mole fractions
gpu_x = gpu_result.X.sel(vertex=0).values
print(f"  X(NB) = {gpu_x[0]:.6f}")
print(f"  X(TI) = {gpu_x[1]:.6f}")

print("\n=== Analysis ===")
print(f"Energy difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")
print(f"Site fraction difference Y(NB): {abs(gpu_y[0] - cpu_y[0]):.6f}")
print(f"Site fraction difference Y(TI): {abs(gpu_y[1] - cpu_y[1]):.6f}")