#!/usr/bin/env python
"""Check GPU convergence behavior for X(TI)=0.9 conditions."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the X(TI)=0.9, T=600K condition that shows 19 J/mol error
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("CHECKING GPU CONVERGENCE FOR X(TI)=0.9, T=600K")
print("=" * 60)

# Run with verbose to see convergence info
print("\nRunning GPU calculation with verbose output...")
print("Look for:")
print("1. ALLOWED_MASS_RESIDUAL value")
print("2. Convergence criteria values")
print("3. Number of iterations")
print("4. Final residuals")
print("\n" + "-" * 60)

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

print("\n" + "-" * 60)
gpu_gm = result_gpu.GM.values.flatten()[0]
print(f"\nGPU Final GM: {gpu_gm:.6f} J/mol")

# Now run CPU for comparison
print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
print(f"CPU Final GM: {cpu_gm:.6f} J/mol")

error = abs(cpu_gm - gpu_gm)
print(f"\nError: {error:.6f} J/mol")

# Extract key info from output
print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("1. Check if GPU reaches 'converged = 1.0'")
print("2. Check final mass_residual value")
print("3. Check number of iterations before convergence")
print("4. Check if convergence criteria are too loose")

# Save analysis
with open('/tmp/convergence_analysis.txt', 'w') as f:
    f.write(f"X(TI)=0.9, T=600K Convergence Analysis\n")
    f.write("=" * 60 + "\n")
    f.write(f"CPU GM: {cpu_gm:.6f} J/mol\n")
    f.write(f"GPU GM: {gpu_gm:.6f} J/mol\n")
    f.write(f"Error: {error:.6f} J/mol\n")
    f.write("\nExpected error: < 0.001 J/mol\n")
    f.write("Actual error: {:.1f}x too large\n".format(error / 0.001))
    f.write("\nLook for these issues in verbose output:\n")
    f.write("1. Premature convergence (converged=1.0 too early)\n")
    f.write("2. Mass residual tolerance too loose\n")
    f.write("3. Phase amount/composition tolerances too loose\n")
    f.write("4. Incorrect convergence logic\n")

print("\nAnalysis saved to /tmp/convergence_analysis.txt")