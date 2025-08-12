#!/usr/bin/env python
"""Debug the initial state mole fractions to understand RHS calculation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")
import sys
import io
from contextlib import redirect_stdout
import re

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.2,  # X_prescribed for CU
    v.X('FE'): 0.3   # X_prescribed for FE
}

print("=" * 80)
print("INITIAL STATE MOLE FRACTIONS DEBUG")
print("=" * 80)

# Run GPU with debug output
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_lines = gpu_output.split('\n')

# Find initial state info
print("\nGPU INITIAL STATE:")
print("-" * 40)

# Look for initial compositions
for i, line in enumerate(gpu_lines[:200]):
    if 'Initial mass balance' in line:
        print(f"  {line.strip()}")
    elif 'Initial system mole_fractions' in line:
        print(f"  {line.strip()}")
    elif 'Starting with' in line and 'phases' in line:
        print(f"  {line.strip()}")
    elif 'Phase 0:' in line and i < 100:  # Early in output
        print(f"  {line.strip()}")
        for j in range(1, 5):
            if i+j < len(gpu_lines):
                next_line = gpu_lines[i+j].strip()
                if 'NP=' in next_line or 'phase_amt' in next_line or 'X[' in next_line:
                    print(f"    {next_line}")
    elif 'Phase 1:' in line and i < 100:
        print(f"  {line.strip()}")
        for j in range(1, 5):
            if i+j < len(gpu_lines):
                next_line = gpu_lines[i+j].strip()
                if 'NP=' in next_line or 'phase_amt' in next_line or 'X[' in next_line:
                    print(f"    {next_line}")
    elif 'Phase 2:' in line and i < 100:
        print(f"  {line.strip()}")
        for j in range(1, 5):
            if i+j < len(gpu_lines):
                next_line = gpu_lines[i+j].strip()
                if 'NP=' in next_line or 'phase_amt' in next_line or 'X[' in next_line:
                    print(f"    {next_line}")

# Look for iteration 0 mole fractions
print("\n" + "=" * 80)
print("GPU ITERATION 0 MOLE FRACTIONS:")
print("-" * 40)

found_iter0 = False
for i, line in enumerate(gpu_lines):
    if 'Iteration 0' in line:
        found_iter0 = True
    elif found_iter0 and 'Iteration 1' in line:
        break
    elif found_iter0:
        if 'state->mole_fractions' in line or 'system mole_fractions' in line:
            print(f"  {line.strip()}")
        elif 'MOLE FRAC CONSTRAINT' in line:
            print(f"  {line.strip()}")
            # Get next line for actual values
            if i+1 < len(gpu_lines) and 'mole_fractions' in gpu_lines[i+1]:
                print(f"    {gpu_lines[i+1].strip()}")

# Now let's manually calculate what the RHS should be
print("\n" + "=" * 80)
print("MANUAL CALCULATION:")
print("-" * 40)

print("\nBased on initial phase compositions:")
print("  Phase 0: NP=0.037, X(CU)=0.118, X(FE)=0.346")
print("  Phase 1: NP=0.305, X(CU)=0.403, X(FE)=0.120")
print("  Phase 2: NP=0.658, X(CU)=0.111, X(FE)=0.381")

# Calculate system mole fractions
phase_amounts = [0.037, 0.305, 0.658]
cu_fractions = [0.118, 0.403, 0.111]
fe_fractions = [0.346, 0.120, 0.381]

system_cu = sum(phase_amounts[i] * cu_fractions[i] for i in range(3))
system_fe = sum(phase_amounts[i] * fe_fractions[i] for i in range(3))

print(f"\nSystem X(CU) = {system_cu:.4f}")
print(f"System X(FE) = {system_fe:.4f}")

print(f"\nPrescribed X(CU) = 0.2000")
print(f"Prescribed X(FE) = 0.3000")

print(f"\nRHS for X(CU) constraint = X_prescribed - X_current = 0.2000 - {system_cu:.4f} = {0.2 - system_cu:.4f}")
print(f"RHS for X(FE) constraint = X_prescribed - X_current = 0.3000 - {system_fe:.4f} = {0.3 - system_fe:.4f}")

print("\n" + "=" * 80)
print("COMPARISON:")
print("-" * 40)
print(f"CPU RHS values: X(CU)=-0.038, X(FE)=+0.025")
print(f"GPU RHS values: X(CU)=-0.136, X(FE)=+0.122")
print(f"Manual calc:    X(CU)={0.2 - system_cu:.3f}, X(FE)={0.3 - system_fe:.3f}")

print("\nThe GPU RHS values don't match our manual calculation!")
print("This suggests the GPU is using different initial phase amounts or compositions.")