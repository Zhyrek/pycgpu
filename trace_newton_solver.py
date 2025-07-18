#!/usr/bin/env python
"""Trace Newton solver behavior to understand why GPU isn't converging."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test X(TI)=0.9, T=600K which shows poor convergence
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("TRACING GPU NEWTON SOLVER BEHAVIOR")
print("=" * 60)
print("Target: X(TI) = 0.9, T = 600K")
print("\nRunning GPU calculation with verbose output...")
print("Focus on:")
print("1. Mass residual at each iteration")
print("2. Whether phase amounts are changing")
print("3. Site fraction updates (delta_y)")
print("4. Convergence criteria values")
print("\n" + "-" * 60)

# Run GPU with verbose to see detailed iteration info
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

print("\n" + "-" * 60)
print("\nKEY OBSERVATIONS:")

# Extract final values
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()

# Calculate actual X(TI)
overall_x_ti = 0.0
for i, np_val in enumerate(gpu_np):
    if np_val > 1e-12:
        overall_x_ti += np_val * gpu_x_ti[i]

print(f"\nFinal GPU X(TI): {overall_x_ti:.6f} (target: 0.900000)")
print(f"Final GPU GM: {gpu_gm:.6f} J/mol")

# Compare with CPU
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"\nCPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {abs(cpu_gm - gpu_gm):.6f} J/mol")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)

# Look for patterns in the verbose output
print("\nCheck the verbose output above for:")
print("1. Is mass_residual decreasing monotonically?")
print("2. Are phase amounts (NP) changing significantly?")  
print("3. Are site fractions (Y) updating properly?")
print("4. Is the solver hitting iteration limit (200)?")
print("5. Are convergence tolerances appropriate?")

print("\nPossible issues to investigate:")
print("1. Step size control - is it too conservative?")
print("2. Matrix conditioning - is the equilibrium matrix ill-conditioned?")
print("3. Constraint enforcement - is the mass balance row correctly formed?")
print("4. Newton update calculation - are delta values computed correctly?")

# Save detailed trace for analysis
with open('/tmp/newton_trace.txt', 'w') as f:
    f.write(f"GPU Newton Solver Trace for X(TI)=0.9, T=600K\n")
    f.write("=" * 60 + "\n")
    f.write(f"Target X(TI): 0.900000\n")
    f.write(f"Actual X(TI): {overall_x_ti:.6f}\n")
    f.write(f"Residual: {abs(overall_x_ti - 0.9):.6f}\n")
    f.write(f"Energy error: {abs(cpu_gm - gpu_gm):.6f} J/mol\n")
    f.write("\nLook for these patterns in verbose output:\n")
    f.write("- Mass residual should decrease each iteration\n")
    f.write("- Convergence should occur before iteration 200\n")
    f.write("- Final mass_residual should be < 1e-12\n")

print("\nTrace saved to /tmp/newton_trace.txt")