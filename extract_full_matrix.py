#!/usr/bin/env python
"""Extract the full equilibrium matrix from CPU and GPU for iteration 0."""

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
print("FULL EQUILIBRIUM MATRIX - ITERATION 0")
print("=" * 80)

# Run GPU and capture output
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()
gpu_lines = gpu_output.split('\n')

# Look for GPU matrix elements
print("\nGPU EQUILIBRIUM MATRIX ELEMENTS:")
print("-" * 40)

# Find where iteration 0 matrix construction happens
for i, line in enumerate(gpu_lines):
    if '[GPU MATRIX CONSTRUCTION] Iteration 0' in line:
        print(f"Found at line {i}: {line}")
        # Print next 50 lines to capture matrix
        for j in range(i, min(i+50, len(gpu_lines))):
            print(gpu_lines[j])
        break

# Look for specific matrix element patterns
print("\n" + "=" * 80)
print("GPU MATRIX ELEMENTS (searching for patterns):")
print("-" * 40)

# Search for matrix row/column entries
matrix_patterns = [
    r'e_matrix\[.*\].*=',
    r'A\[.*\].*=',
    r'matrix\[.*\].*=',
    r'row.*col.*=',
    r'Matrix element',
    r'eq_matrix',
    r'equilibrium_matrix',
    r'\[GPU MATRIX\]',
    r'Matrix row',
    r'Matrix col',
    r'c_component\[.*\]',
    r'e_matrix\[.*\]',
    r'constraint_matrix',
    r'Constraint row',
    r'Phase.*row',
    r'eq_soln\[.*\]'
]

found_matrix_lines = []
for line in gpu_lines:
    for pattern in matrix_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            found_matrix_lines.append(line.strip())
            break

print(f"Found {len(found_matrix_lines)} matrix-related lines")
for line in found_matrix_lines[:100]:  # Show first 100
    print(f"  {line}")

# Look for the solver input/output
print("\n" + "=" * 80)
print("GPU SOLVER INPUT/OUTPUT:")
print("-" * 40)

# Find solve_equilibrium lines
for i, line in enumerate(gpu_lines):
    if 'solve_equilibrium' in line.lower() and 'iteration 0' in line.lower():
        print(f"Found solve call at line {i}")
        for j in range(max(0, i-5), min(i+20, len(gpu_lines))):
            print(gpu_lines[j])
        break

# Look for LU decomposition or solve steps
print("\n" + "=" * 80)
print("GPU LINEAR SOLVER DETAILS:")
print("-" * 40)

solver_patterns = [
    r'LU decomposition',
    r'lu_solve',
    r'solve_linear',
    r'gaussian',
    r'pivot',
    r'forward substitution',
    r'backward substitution',
    r'solution vector',
    r'delta.*=',
    r'solver result',
    r'linear solve'
]

solver_lines = []
for line in gpu_lines:
    for pattern in solver_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            solver_lines.append(line.strip())
            break

print(f"Found {len(solver_lines)} solver-related lines")
for line in solver_lines[:50]:
    print(f"  {line}")

# Now look for CPU matrix
print("\n" + "=" * 80)
print("CPU EQUILIBRIUM MATRIX STRUCTURE:")
print("-" * 40)

# Run CPU with special debug flags
import os
os.environ['PYCALPHAD_DEBUG_MATRIX'] = '1'

cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_lines = cpu_output.split('\n')

# Extract CPU matrix structure
print("\nCPU Matrix at Iteration 0:")
in_iter0 = False
for i, line in enumerate(cpu_lines):
    if 'state.iteration=0' in line:
        in_iter0 = True
        print(f"Starting iteration 0 at line {i}")
    elif in_iter0 and 'state.iteration=1' in line:
        in_iter0 = False
        break
    elif in_iter0:
        if any(x in line for x in ['matrix', 'Matrix', 'row', 'col', 'RHS', 'equilibrium', 'constraint']):
            print(f"  {line.strip()}")

# Summary
print("\n" + "=" * 80)
print("MATRIX STRUCTURE SUMMARY:")
print("-" * 40)
print("\nExpected structure for 3 phases, 2 composition constraints:")
print("  Columns: [μ₁, μ₂, NP₁, NP₂, NP₃]")
print("  Row 1: Energy balance")
print("  Row 2: Mass balance AL")
print("  Row 3: Mass balance CU")
print("  Row 4: Mass balance FE")
print("  Row 5: Constraint X(CU)")
print("  Row 6: Constraint X(FE)")
print("\nMatrix dimensions: 6x5")
print("\nThe solution vector gives changes in:")
print("  δμ₁, δμ₂ (chemical potentials)")
print("  δNP₁, δNP₂, δNP₃ (phase amounts)")