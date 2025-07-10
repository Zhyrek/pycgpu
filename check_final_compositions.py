#!/usr/bin/env python3
"""Check final compositions from CPU and GPU"""
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

print("=== Final Composition Check ===")

# Run CPU calculation
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
cpu_y = cpu_result.Y.values
cpu_x = cpu_result.X.values

print("CPU Results:")
print(f"  GM: {cpu_gm:.1f} J/mol")
print(f"  Y shape: {cpu_y.shape}")
# Find active phase
cpu_np = cpu_result.NP.values.flatten()
active_idx = np.where(cpu_np > 1e-10)[0]
if len(active_idx) > 0:
    idx = active_idx[0]
    print(f"  Active phase index: {idx}")
    # Y array has shape (phases, sublattices, sites_per_sublattice, ..., components)
    # For BCC_A2 with 1 sublattice and 2 components (NB, TI)
    print(f"  Y(BCC_A2,0,NB) = {float(cpu_y.flatten()[0]):.6f}")
    print(f"  Y(BCC_A2,0,TI) = {float(cpu_y.flatten()[1]):.6f}")
    print(f"  X(NB) = {float(cpu_x.flatten()[0]):.6f}")
    print(f"  X(TI) = {float(cpu_x.flatten()[1]):.6f}")

# Run GPU calculation
gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])
gpu_y = gpu_result.Y.values
gpu_x = gpu_result.X.values

print("\nGPU Results:")
print(f"  GM: {gpu_gm:.1f} J/mol")
print(f"  Y shape: {gpu_y.shape}")
# Find active phase
gpu_np = gpu_result.NP.values.flatten()
active_idx = np.where(gpu_np > 1e-10)[0]
if len(active_idx) > 0:
    idx = active_idx[0]
    print(f"  Active phase index: {idx}")
    # Y array has shape (phases, sublattices, sites_per_sublattice, ..., components)
    # For BCC_A2 with 1 sublattice and 2 components (NB, TI)
    print(f"  Y(BCC_A2,0,NB) = {float(gpu_y.flatten()[0]):.6f}")
    print(f"  Y(BCC_A2,0,TI) = {float(gpu_y.flatten()[1]):.6f}")
    print(f"  X(NB) = {float(gpu_x.flatten()[0]):.6f}")
    print(f"  X(TI) = {float(gpu_x.flatten()[1]):.6f}")

print("\n=== Analysis ===")
print("The CPU consolidates 2 BCC_A2 phases at iteration 1")
print("The GPU removes the second phase at iteration 0")
print("This leads to different convergence paths and final compositions")