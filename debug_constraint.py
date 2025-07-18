#!/usr/bin/env python
"""Debug constraint handling in GPU."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test X(TI) = 0.005
conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

print("Testing X(TI) = 0.005 constraint...")

# CPU first
print("\n=== CPU ===")
result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=False)
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
print(f"CPU X(TI) = {cpu_x_ti:.8f}")

# GPU with minimal output
print("\n=== GPU ===")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"GPU X(TI) = {gpu_x_ti:.8f}")

print(f"\nDifference: {abs(cpu_x_ti - gpu_x_ti):.6f}")