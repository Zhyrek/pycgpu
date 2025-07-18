#!/usr/bin/env python
"""Debug why GPU has high mass residual for X(TI)=0.9."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("DEBUGGING GPU MASS RESIDUAL")
print("=" * 60)

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

# Check actual composition
gpu_x_ti_per_phase = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print("\nGPU Results:")
print(f"Phase fractions: {gpu_np}")
print(f"X(TI) per phase: {gpu_x_ti_per_phase}")

# Calculate overall X(TI)
overall_x_ti = 0.0
total_amount = 0.0
for i, np_val in enumerate(gpu_np):
    if np_val > 1e-12:
        overall_x_ti += np_val * gpu_x_ti_per_phase[i]
        total_amount += np_val
        print(f"  Phase {i}: NP={np_val:.6f}, X(TI)={gpu_x_ti_per_phase[i]:.6f}")

if total_amount > 0:
    overall_x_ti /= total_amount
    
print(f"\nOverall X(TI) = {overall_x_ti:.6f}")
print(f"Target X(TI) = 0.900000")
print(f"Residual = {abs(overall_x_ti - 0.9):.6f}")

# Compare with CPU
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_x_ti_per_phase = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print("\nCPU Results:")
print(f"Phase fractions: {cpu_np}")
print(f"X(TI) per phase: {cpu_x_ti_per_phase}")

# Calculate CPU overall X(TI)
cpu_overall_x_ti = 0.0
cpu_total_amount = 0.0
for i, np_val in enumerate(cpu_np):
    if np_val > 1e-12:
        cpu_overall_x_ti += np_val * cpu_x_ti_per_phase[i]
        cpu_total_amount += np_val
        print(f"  Phase {i}: NP={np_val:.6f}, X(TI)={cpu_x_ti_per_phase[i]:.6f}")

if cpu_total_amount > 0:
    cpu_overall_x_ti /= cpu_total_amount
    
print(f"\nOverall X(TI) = {cpu_overall_x_ti:.6f}")
print(f"Residual = {abs(cpu_overall_x_ti - 0.9):.6f}")

# Energy comparison
gpu_gm = result_gpu.GM.values.flatten()[0]
cpu_gm = result_cpu.GM.values.flatten()[0]

print(f"\nEnergy comparison:")
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"Error: {abs(cpu_gm - gpu_gm):.6f} J/mol")

# Analysis
print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)
if abs(overall_x_ti - 0.9) > 0.001:
    print("✗ GPU FAILED to satisfy composition constraint!")
    print(f"  GPU X(TI) = {overall_x_ti:.6f} instead of 0.900000")
    print(f"  This explains the {abs(cpu_gm - gpu_gm):.1f} J/mol energy error")
    print("\nPossible causes:")
    print("1. Newton solver not updating properly")
    print("2. Constraint equation not being enforced correctly")
    print("3. Convergence check allowing premature termination")
else:
    print("✓ GPU satisfied composition constraint")

with open('/tmp/residual_analysis.txt', 'w') as f:
    f.write(f"GPU X(TI) = {overall_x_ti:.6f} (target: 0.9)\n")
    f.write(f"CPU X(TI) = {cpu_overall_x_ti:.6f} (target: 0.9)\n")
    f.write(f"GPU residual: {abs(overall_x_ti - 0.9):.6f}\n")
    f.write(f"CPU residual: {abs(cpu_overall_x_ti - 0.9):.6f}\n")