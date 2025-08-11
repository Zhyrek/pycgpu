#!/usr/bin/env python
"""Debug a single ternary condition to find exact divergence point."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition that's failing
conditions = {
    v.T: 800,
    v.P: 101325,
    v.X('CU'): 0.1,
    v.X('FE'): 0.1  # X(AL) = 0.8 implied
}

print("=" * 60)
print("DEBUGGING SINGLE TERNARY CONDITION")
print("=" * 60)
print(f"T = 800 K")
print(f"X(CU) = 0.1, X(FE) = 0.1, X(AL) = 0.8")
print()

# Run CPU calculation
print("CPU Calculation:")
print("-" * 40)
result_cpu = equilibrium(dbf, comps, phases, conditions, calc_opts={'pdens': 50})
cpu_gm = float(result_cpu.GM.values)
cpu_mu = result_cpu.MU.values[0,0,0,0,0,:]
cpu_np = result_cpu.NP.values[0,0,0,0,0,:]

print(f"  GM = {cpu_gm:.2f} J/mol")
print(f"  MU: AL={cpu_mu[0]:.2f}, CU={cpu_mu[1]:.2f}, FE={cpu_mu[2]:.2f}")
print(f"  Phase amounts: {cpu_np}")

# Run GPU calculation with verbose to see what's happening
print("\nGPU Calculation (verbose):")
print("-" * 40)

import sys
import io
from contextlib import redirect_stdout

# Capture verbose output
captured = io.StringIO()
with redirect_stdout(captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

output = captured.getvalue()

# Look for key information in verbose output
print("\nKey GPU debug lines:")
for line in output.split('\n'):
    # Look for constraint-related info
    if 'constraint' in line.lower() or 'prescribed_mole' in line.lower():
        print(f"  {line.strip()}")
    # Look for initial values
    elif 'initial_chemical_potentials' in line:
        print(f"  {line.strip()}")
    # Look for convergence info
    elif 'converged' in line.lower():
        print(f"  {line.strip()}")
    # Look for matrix info
    elif 'equilibrium_matrix' in line.lower():
        print(f"  {line.strip()}")

gpu_gm = float(result_gpu.GM.values)
gpu_mu = result_gpu.MU.values[0,0,0,0,0,:]
gpu_np = result_gpu.NP.values[0,0,0,0,0,:]

print(f"\nGPU Results:")
print(f"  GM = {gpu_gm:.2f} J/mol")
print(f"  MU: AL={gpu_mu[0]:.2f}, CU={gpu_mu[1]:.2f}, FE={gpu_mu[2]:.2f}")
print(f"  Phase amounts: {gpu_np}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
gm_diff = abs(cpu_gm - gpu_gm)
print(f"GM Difference: {gm_diff:.2f} J/mol")
print(f"MU Differences:")
print(f"  AL: {abs(cpu_mu[0] - gpu_mu[0]):.2f}")
print(f"  CU: {abs(cpu_mu[1] - gpu_mu[1]):.2f}")
print(f"  FE: {abs(cpu_mu[2] - gpu_mu[2]):.2f}")

if gm_diff > 100:
    print("\n✗ SIGNIFICANT DIVERGENCE")
    print("\nPossible causes:")
    print("1. Constraint matrix setup differs for ternary systems")
    print("2. Mole fraction constraint RHS values are wrong")
    print("3. Number of free chemical potentials is incorrect")
    print("4. Constraint coefficient matrix has wrong values")
else:
    print("\n✓ Results match within tolerance")