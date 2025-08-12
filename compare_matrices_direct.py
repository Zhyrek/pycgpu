#!/usr/bin/env python
"""Direct comparison of CPU and GPU equilibrium matrices."""

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
    v.X('CU'): 0.2,
    v.X('FE'): 0.3  # X(AL) = 0.5 implied
}

print("=" * 80)
print("EQUILIBRIUM MATRIX COMPARISON - ITERATION 0")
print("=" * 80)

# Run GPU and extract matrix
gpu_captured = io.StringIO()
with redirect_stdout(gpu_captured):
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True, calc_opts={'pdens': 50})

gpu_output = gpu_captured.getvalue()

# Extract GPU matrix
print("\nGPU EQUILIBRIUM MATRIX (Iteration 0):")
print("-" * 40)

gpu_matrix_found = False
for line in gpu_output.split('\n'):
    if '[GPU EQUILIBRIUM MATRIX] Complete matrix at iteration 0' in line:
        gpu_matrix_found = True
    elif gpu_matrix_found and 'Row' in line:
        print(line)
        if 'Row 5:' in line:
            # Get the solution line too
            break

# Extract solution
for line in gpu_output.split('\n'):
    if 'RHS after lstsq (solution):' in line:
        print(f"\nGPU Solution vector: {line.split(':')[1].strip()}")
        break

# Run CPU and extract information
print("\n" + "=" * 80)
print("CPU EQUILIBRIUM MATRIX (Iteration 0):")
print("-" * 40)

cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_lines = cpu_output.split('\n')

# Extract CPU energy balance rows
print("\nCPU Energy Balance (from debug output):")
energy_rows = []
for i, line in enumerate(cpu_lines[:200]):  # First 200 lines should have initial matrix
    if 'Writing row for stable phase:' in line and 'iteration=0' in ' '.join(cpu_lines[max(0,i-10):i]):
        # Found a row, extract values
        for j in range(1, 5):
            if i+j < len(cpu_lines):
                next_line = cpu_lines[i+j]
                if 'Energy:' in next_line:
                    energy = re.search(r'[-\d.e+]+', next_line.split('Energy:')[1])
                    if energy:
                        print(f"  Energy: {energy.group()}")
                elif 'Masses:' in next_line:
                    print(f"  {next_line.strip()}")
                elif 'Row values' in next_line:
                    print(f"  {next_line.strip()}")
                elif 'RHS:' in next_line and 'out_rhs' not in next_line:
                    print(f"  {next_line.strip()}")

# Extract CPU constraint information
print("\nCPU Mole Fraction Constraints (Iteration 0):")
constraint_info = []
in_iter0 = False
for i, line in enumerate(cpu_lines):
    if 'state.iteration=0' in line:
        in_iter0 = True
    elif in_iter0 and 'state.iteration=1' in line:
        in_iter0 = False
        break
    elif in_iter0 and 'write_row_fixed_mole_fraction' in line:
        # Extract the constraint
        phase = re.search(r'phase_idx=(\d+)', line)
        comp = re.search(r'component_idx=(\d+)', line)
        if phase and comp:
            phase_idx = int(phase.group(1))
            comp_idx = int(comp.group(1))
            # Get RHS from next line
            if i+1 < len(cpu_lines) and 'out_rhs' in cpu_lines[i+1]:
                rhs = re.search(r'entry:\s*([-\d.e+]+)', cpu_lines[i+1])
                if rhs:
                    rhs_val = float(rhs.group(1))
                    constraint_info.append((phase_idx, comp_idx, rhs_val))

# Group by component
cu_constraints = [(p, r) for p, c, r in constraint_info if c == 1]
fe_constraints = [(p, r) for p, c, r in constraint_info if c == 2]

print("\nX(CU) constraints (component 1):")
for phase, rhs in cu_constraints:
    print(f"  Phase {phase}: RHS = {rhs:.6f}")

print("\nX(FE) constraints (component 2):")
for phase, rhs in fe_constraints:
    print(f"  Phase {phase}: RHS = {rhs:.6f}")

# Calculate total RHS for constraints
if cu_constraints:
    total_cu_rhs = sum(r for _, r in cu_constraints)
    print(f"\nTotal X(CU) RHS: {total_cu_rhs:.6f}")

if fe_constraints:
    total_fe_rhs = sum(r for _, r in fe_constraints)
    print(f"Total X(FE) RHS: {total_fe_rhs:.6f}")

# COMPARISON
print("\n" + "=" * 80)
print("KEY DIFFERENCES:")
print("-" * 40)

print("\n1. CONSTRAINT RHS VALUES:")
print("   CPU X(CU) total error: ~", sum(abs(r) for _, r in cu_constraints))
print("   CPU X(FE) total error: ~", sum(abs(r) for _, r in fe_constraints))
print("   GPU Row 3 RHS: -0.136 (much larger!)")
print("   GPU Row 4 RHS: +0.122 (much larger!)")

print("\n2. SOLUTION VECTOR:")
print("   GPU: [-147000, -25700, +26.8, -1.22, -25.6]")
print("   This produces:")
print("     Phase 0: +26.8 change (2680% increase!)")
print("     Phase 1: -1.22 change")
print("     Phase 2: -25.6 change (eliminated!)")

print("\n3. THE PROBLEM:")
print("   The GPU constraint RHS values are ~10x larger than CPU")
print("   This causes the solver to produce huge changes")
print("   The system becomes unstable and can't converge")

# Show what the CPU does
print("\n4. CPU BEHAVIOR (from iteration data):")
after_iter1 = False
for i, line in enumerate(cpu_lines):
    if 'state.iteration=1' in line:
        after_iter1 = True
    elif after_iter1 and 'write_row_fixed_mole_fraction' in line:
        # Show the RHS values after iteration 1
        if i+1 < len(cpu_lines) and 'out_rhs' in cpu_lines[i+1]:
            rhs = re.search(r'entry:\s*([-\d.e+]+)', cpu_lines[i+1])
            if rhs and abs(float(rhs.group(1))) > 1e-10:
                print(f"   After iter 1: {cpu_lines[i].strip()}")
                print(f"                 {cpu_lines[i+1].strip()}")
                break