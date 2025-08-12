#!/usr/bin/env python
"""Debug the constraint RHS values to understand why GPU values are 10x larger than CPU."""

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
# This implies X(AL) = 0.5

print("=" * 80)
print("CONSTRAINT RHS VALUE ANALYSIS")
print("=" * 80)

print("\nPRESCRIBED VALUES:")
print(f"  X(CU) prescribed = 0.2")
print(f"  X(FE) prescribed = 0.3")
print(f"  X(AL) implied = 0.5")

# Run GPU and extract constraint info
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_lines = gpu_output.split('\n')

# Find initial state mole fractions
print("\n" + "=" * 80)
print("GPU INITIAL STATE")
print("=" * 80)

# Look for initial mole fractions
for i, line in enumerate(gpu_lines[:100]):  # First 100 lines
    if 'Initial state mole_fractions' in line or 'state->mole_fractions' in line:
        print(f"  {line.strip()}")
    elif 'initial_mole_fractions' in line:
        print(f"  {line.strip()}")

# Look for constraint residual calculations
print("\n" + "=" * 80)
print("GPU CONSTRAINT RESIDUAL CALCULATIONS (Iteration 0)")
print("=" * 80)

found_iter0 = False
for i, line in enumerate(gpu_lines):
    if 'Iteration 0' in line:
        found_iter0 = True
    elif found_iter0 and 'Iteration 1' in line:
        break
    elif found_iter0:
        if 'MOLE FRAC CONSTRAINT' in line:
            print(f"  {line.strip()}")
            # Get next line for mole fractions
            if i+1 < len(gpu_lines) and 'mole_fractions' in gpu_lines[i+1]:
                print(f"  {gpu_lines[i+1].strip()}")
        elif 'component_residual' in line:
            print(f"  {line.strip()}")
        elif 'RHS before residual' in line or 'RHS after' in line:
            print(f"  {line.strip()}")

# Look for the actual constraint matrix values
print("\n" + "=" * 80)
print("GPU EQUILIBRIUM MATRIX - CONSTRAINT ROWS")
print("=" * 80)

for line in gpu_lines:
    if '[GPU EQUILIBRIUM MATRIX] Complete matrix at iteration 0' in line:
        idx = gpu_lines.index(line)
        # Get rows 3 and 4 (constraint rows)
        for j in range(idx, min(idx+10, len(gpu_lines))):
            if 'Row 3:' in gpu_lines[j] or 'Row 4:' in gpu_lines[j]:
                print(f"  {gpu_lines[j].strip()}")

# Run CPU and extract similar info
print("\n" + "=" * 80)
print("CPU INITIAL STATE")
print("=" * 80)

cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_lines = cpu_output.split('\n')

# Find CPU initial mole fractions
for line in cpu_lines[:200]:
    if 'initial_mole_fractions' in line or 'state.mole_fractions' in line:
        print(f"  {line.strip()}")

# Find CPU constraint RHS values
print("\n" + "=" * 80)
print("CPU CONSTRAINT RHS VALUES (Iteration 0)")
print("=" * 80)

in_iter0 = False
for i, line in enumerate(cpu_lines):
    if 'state.iteration=0' in line:
        in_iter0 = True
    elif in_iter0 and 'state.iteration=1' in line:
        break
    elif in_iter0 and 'write_row_fixed_mole_fraction' in line:
        print(f"  {line.strip()}")
        # Get RHS from next line
        if i+1 < len(cpu_lines) and 'out_rhs' in cpu_lines[i+1]:
            print(f"    {cpu_lines[i+1].strip()}")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nKEY FORMULA:")
print("  component_residual = X_current - X_prescribed")
print("  equilibrium_rhs = -(component_residual) = X_prescribed - X_current")

print("\nFor X(CU) = 0.2:")
print("  If X_current(CU) = 0.063 initially (from phase compositions)")
print("  Then RHS = 0.2 - 0.063 = 0.137")
print("  This matches GPU Row 3 RHS: -0.136!")

print("\nFor X(FE) = 0.3:")
print("  If X_current(FE) = 0.178 initially")
print("  Then RHS = 0.3 - 0.178 = 0.122")
print("  This matches GPU Row 4 RHS: +0.122!")

print("\n*** THE PROBLEM ***")
print("The GPU is using the WRONG sign for the constraint RHS!")
print("It appears to be using X_current - X_prescribed instead of X_prescribed - X_current")
print("Or there's a sign flip somewhere in the matrix construction.")