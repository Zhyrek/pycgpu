#!/usr/bin/env python
"""Quick test for binary system."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC', 'RHOMBO']

# Single binary condition
conditions = {
    v.T: 600,
    v.P: 101325,
    v.X('BI'): 0.1
}

print("=" * 60)
print("BINARY SYSTEM TEST")
print("=" * 60)
print(f"Components: {comps}")
print(f"T = {conditions[v.T]} K, X(BI) = {conditions[v.X('BI')]}, X(AU) = {1.0 - conditions[v.X('BI')]}")
print()

# CPU calculation
print("CPU Calculation:")
result_cpu = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)
print(f"  GM = {cpu_gm:.2f} J/mol")
print()

# GPU calculation
print("GPU Calculation:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(result_gpu.GM.values)
print(f"  GM = {gpu_gm:.2f} J/mol")
print()

diff = abs(cpu_gm - gpu_gm)
print(f"Difference: {diff:.2f} J/mol")

if diff < 1.0:
    print("✓ Results match")
else:
    print("✗ Results differ")