#!/usr/bin/env python
"""Compare the equilibrium matrix for the first solver step between CPU and GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import re

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
print("EQUILIBRIUM MATRIX COMPARISON - FIRST SOLVER STEP")
print("=" * 80)

# Run CPU and capture output
cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()

# Run GPU and capture output  
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()

# Parse CPU matrix information
print("\n" + "=" * 80)
print("CPU EQUILIBRIUM MATRIX - ITERATION 0")
print("=" * 80)

# Find CPU iteration 0 matrix construction
cpu_lines = cpu_output.split('\n')
in_iter0 = False
matrix_lines = []
for i, line in enumerate(cpu_lines):
    if 'construct_equilibrium_system called, state.iteration=0' in line:
        in_iter0 = True
    elif in_iter0 and 'construct_equilibrium_system called, state.iteration=1' in line:
        in_iter0 = False
        break
    elif in_iter0:
        matrix_lines.append(line)

# Extract CPU matrix values
print("\nCPU Matrix Construction Details:")
for line in matrix_lines:
    if 'num_stable_phases' in line:
        print(f"  {line.strip()}")
    elif 'num_fixed_mole_fraction_conditions' in line:
        print(f"  {line.strip()}")
    elif 'write_row_fixed_mole_fraction' in line:
        print(f"  {line.strip()}")
    elif 'out_rhs' in line:
        print(f"    {line.strip()}")

# Look for CPU equilibrium matrix rows
print("\nCPU Equilibrium Matrix Rows (from earlier output):")
for line in cpu_lines[:500]:  # Check first 500 lines for initial matrix
    if 'Writing row for stable phase:' in line:
        idx = cpu_lines.index(line)
        print(f"  {line.strip()}")
        # Get next few lines for details
        for j in range(1, 5):
            if idx+j < len(cpu_lines):
                next_line = cpu_lines[idx+j].strip()
                if 'Energy:' in next_line or 'Masses:' in next_line or 'Row values' in next_line or 'RHS:' in next_line:
                    print(f"    {next_line}")

# Parse GPU matrix information
print("\n" + "=" * 80)
print("GPU EQUILIBRIUM MATRIX - ITERATION 0")
print("=" * 80)

gpu_lines = gpu_output.split('\n')

# Find GPU iteration 0 matrix info
print("\nGPU Matrix Construction Details:")
for line in gpu_lines:
    if 'Iteration 0/200' in line:
        print(f"Found iteration 0 start: {line.strip()}")
        break

# Look for matrix dimensions
for line in gpu_lines:
    if '[GPU MATRIX SIZE] Iteration 0:' in line:
        print(f"  {line.strip()}")
        idx = gpu_lines.index(line)
        if idx+1 < len(gpu_lines):
            print(f"  {gpu_lines[idx+1].strip()}")
        break

# Look for GPU constraint setup
print("\nGPU Constraint Values:")
for line in gpu_lines:
    if 'Coefficients:' in line and 'Thread 0' not in line:
        print(f"  {line.strip()}")
        break
    elif 'GPU DEBUG: Coefficients:' in line:
        print(f"  {line.strip()}")
        break

for line in gpu_lines:
    if 'RHS:' in line and 'Thread 0' not in line and 'out_rhs' not in line:
        print(f"  {line.strip()}")
        break
    elif 'GPU DEBUG: RHS:' in line:
        print(f"  {line.strip()}")
        break

# Look for phase information at iteration 0
print("\nGPU Phase Information at Iteration 0:")
found_iter0 = False
phase_info_lines = []
for i, line in enumerate(gpu_lines):
    if 'Iteration 0/200' in line or 'Iteration 0:' in line:
        found_iter0 = True
    elif found_iter0 and ('Iteration 1/' in line or 'Iteration 1:' in line):
        break
    elif found_iter0:
        if 'phase_0:' in line or 'phase_1:' in line or 'phase_2:' in line:
            phase_info_lines.append(line.strip())
        elif 'NP=' in line and 'phase' in line.lower():
            phase_info_lines.append(line.strip())
        elif 'phase_amt' in line:
            phase_info_lines.append(line.strip())

for line in phase_info_lines[:10]:
    print(f"  {line}")

# Look for mass balance info
print("\nGPU Mass Balance Information:")
for line in gpu_lines:
    if '[GPU MASS BALANCE]' in line and 'iteration 0:' in line:
        print(f"  {line.strip()}")
        break

# Look for the actual matrix solve
print("\n" + "=" * 80)
print("MATRIX SOLVE RESULTS")
print("=" * 80)

# Find GPU solve results
print("\nGPU Solver Step (Iteration 0 -> 1):")
for line in gpu_lines:
    if 'Phase 0: old=' in line:
        print(f"  {line.strip()}")
        idx = gpu_lines.index(line)
        # Get next two phase lines
        if idx+1 < len(gpu_lines) and 'Phase 1: old=' in gpu_lines[idx+1]:
            print(f"  {gpu_lines[idx+1].strip()}")
        if idx+2 < len(gpu_lines) and 'Phase 2: old=' in gpu_lines[idx+2]:
            print(f"  {gpu_lines[idx+2].strip()}")
        break

# Compare with expected CPU behavior
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nKey Differences:")
print("1. CPU iteration 0 matrix:")
print("   - Has 3 stable phases initially")
print("   - Sets up constraints for X(CU) and X(FE)")
print("   - RHS values show small corrections needed")

print("\n2. GPU iteration 0 matrix:")
print("   - Also starts with 3 phases")
print("   - Matrix dimensions: 6x5 (matching CPU structure)")
print("   - But solver produces HUGE changes:")
print("     * Phase 0: 3.7% -> 72.6% (delta = +26.8!)")
print("     * Phase 2: 65.8% -> 0% (completely eliminated!)")

print("\n3. The problem is likely:")
print("   - Different linear solver behavior (LU vs SVD)")
print("   - Missing damping/line search in GPU solver")
print("   - Incorrect scaling or normalization in GPU matrix")
print("   - Wrong constraint formulation for ternary systems")