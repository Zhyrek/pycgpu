#!/usr/bin/env python
"""Test to trace numerical precision differences between CPU and GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import subprocess

# Clear GPU cache
subprocess.run(['rm', '-f', '/home/user/.cache/pycalphad/*.cu'], capture_output=True)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("=== Testing Numerical Precision ===")
print("\nTarget X(TI) = 0.9000000000")
print("\nRunning GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
print(f"\nGPU Result:")
print(f"  X(TI) = {gpu_x_ti:.15f}")
print(f"  Binary representation: {gpu_x_ti.hex()}")
print(f"  Error = {gpu_x_ti - 0.9:.15e}")

print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
print(f"\nCPU Result:")
print(f"  X(TI) = {cpu_x_ti:.15f}")
print(f"  Binary representation: {cpu_x_ti.hex()}")
print(f"  Error = {cpu_x_ti - 0.9:.15e}")

print(f"\nCPU vs GPU difference: {abs(cpu_x_ti - gpu_x_ti):.15e}")

# Check if this is a floating point representation issue
target = 0.9
print(f"\nTarget 0.9:")
print(f"  Binary representation: {target.hex()}")
print(f"  As float64: {np.float64(0.9):.20f}")

# Check what the exact value should be after consolidation
x_ti_after_consolidation = 0.9029604
print(f"\nGPU's stuck value {x_ti_after_consolidation}:")
print(f"  Binary representation: {x_ti_after_consolidation.hex()}")
print(f"  Difference from 0.9: {x_ti_after_consolidation - 0.9:.15e}")