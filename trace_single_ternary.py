#!/usr/bin/env python
"""Trace a single ternary condition in detail to find divergence point."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.2,
    v.X('FE'): 0.3  # X(AL) = 0.5 implied
}

print("=" * 80)
print("DETAILED TRACE: SINGLE TERNARY CONDITION")
print("=" * 80)
print(f"Components: {comps}")
print(f"Non-VA components: {[c for c in comps if c != 'VA']}")
print(f"Phases: {phases}")
print(f"Conditions: T=1200K, X(CU)=0.2, X(FE)=0.3, X(AL)=0.5")
print()

# First run CPU with debug output
print("=" * 80)
print("CPU CALCULATION WITH DEBUG OUTPUT")
print("=" * 80)

# Enable verbose debug for CPU
import os
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_gm = float(result_cpu.GM.values)
cpu_mu = result_cpu.MU.values[0,0,0,0,:]
cpu_np = result_cpu.NP.values[0,0,0,0,:]
cpu_x = result_cpu.X.values[0,0,0,0,:,:]

print("\n" + "=" * 80)
print("CPU FINAL RESULTS")
print("=" * 80)
print(f"GM = {cpu_gm:.6f} J/mol")
print(f"MU: AL={cpu_mu[0]:.2f}, CU={cpu_mu[1]:.2f}, FE={cpu_mu[2]:.2f}")
print(f"Phase amounts: {cpu_np}")
print(f"Phase compositions:")
for i, phase in enumerate(phases):
    if cpu_np[i] > 1e-10:
        print(f"  {phase}: NP={cpu_np[i]:.6f}, X(AL)={cpu_x[i,0]:.6f}, X(CU)={cpu_x[i,1]:.6f}, X(FE)={cpu_x[i,2]:.6f}")

print("\n" + "=" * 80)
print("GPU CALCULATION WITH DEBUG OUTPUT")
print("=" * 80)

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_gm = float(result_gpu.GM.values)
gpu_mu = result_gpu.MU.values[0,0,0,0,:]
gpu_np = result_gpu.NP.values[0,0,0,0,:]
gpu_x = result_gpu.X.values[0,0,0,0,:,:]

print("\n" + "=" * 80)
print("GPU FINAL RESULTS")
print("=" * 80)
print(f"GM = {gpu_gm:.6f} J/mol")
print(f"MU: AL={gpu_mu[0]:.2f}, CU={gpu_mu[1]:.2f}, FE={gpu_mu[2]:.2f}")
print(f"Phase amounts: {gpu_np}")
print(f"Phase compositions:")
for i, phase in enumerate(phases):
    if gpu_np[i] > 1e-10:
        print(f"  {phase}: NP={gpu_np[i]:.6f}, X(AL)={gpu_x[i,0]:.6f}, X(CU)={gpu_x[i,1]:.6f}, X(FE)={gpu_x[i,2]:.6f}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"GM Difference: {abs(cpu_gm - gpu_gm):.6f} J/mol")
print(f"MU Differences:")
print(f"  AL: {abs(cpu_mu[0] - gpu_mu[0]):.2f}")
print(f"  CU: {abs(cpu_mu[1] - gpu_mu[1]):.2f}")
print(f"  FE: {abs(cpu_mu[2] - gpu_mu[2]):.2f}")
print(f"Phase amount differences:")
for i, phase in enumerate(phases):
    print(f"  {phase}: {abs(cpu_np[i] - gpu_np[i]):.6f}")

if abs(cpu_gm - gpu_gm) > 100:
    print("\n✗ SIGNIFICANT DIVERGENCE DETECTED")
else:
    print("\n✓ Results match within tolerance")