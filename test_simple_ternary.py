#!/usr/bin/env python
"""Simple test of ternary system."""

import numpy as np
import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,  # Higher temperature for liquid stability
    v.P: 101325,
    v.X('CU'): 0.2,
    v.X('FE'): 0.3  # X(AL) = 0.5 implied
}

print("=" * 60)
print("TERNARY SYSTEM TEST")
print("=" * 60)
print(f"Components: {comps}")
print(f"T = 1200 K, X(CU) = 0.2, X(FE) = 0.3, X(AL) = 0.5")
print()

# Run CPU calculation
print("CPU Calculation:")
result_cpu = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)
print(f"  GM = {cpu_gm:.2f} J/mol")

# Run GPU calculation
print("\nGPU Calculation:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(result_gpu.GM.values)
print(f"  GM = {gpu_gm:.2f} J/mol")

print(f"\nDifference: {abs(cpu_gm - gpu_gm):.2f} J/mol")
if abs(cpu_gm - gpu_gm) < 100:
    print("✓ Results match!")
else:
    print("✗ Results differ significantly")