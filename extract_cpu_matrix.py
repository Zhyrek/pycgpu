#!/usr/bin/env python
"""Extract the full CPU equilibrium matrix for iteration 0."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import re

# Add debug instrumentation to CPU code
import pycalphad.core.eqsolver as eqsolver_module

# Monkey patch to capture matrix details
original_solve = eqsolver_module.solve_eq_at_conditions

def debug_solve(*args, **kwargs):
    """Wrapper to capture matrix details."""
    print("[CPU MATRIX INTERCEPT] solve_eq_at_conditions called")
    return original_solve(*args, **kwargs)

eqsolver_module.solve_eq_at_conditions = debug_solve

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
print("CPU EQUILIBRIUM MATRIX EXTRACTION")
print("=" * 80)

# Run CPU with maximum debug output
import os
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'
os.environ['PYCALPHAD_MATRIX_DEBUG'] = '1'

cpu_captured = io.StringIO()
with redirect_stdout(cpu_captured):
    result_cpu = equilibrium(dbf, comps, phases, conditions, verbose=True, calc_opts={'pdens': 50})

cpu_output = cpu_captured.getvalue()
cpu_lines = cpu_output.split('\n')

# Find CPU matrix construction for iteration 0
print("\nCPU EQUILIBRIUM MATRIX - ITERATION 0")
print("-" * 40)

# Look for the matrix construction
in_iter0 = False
matrix_rows = []
rhs_values = []

for i, line in enumerate(cpu_lines):
    # Start of iteration 0
    if 'construct_equilibrium_system called, state.iteration=0' in line:
        in_iter0 = True
        print(f"Found iteration 0 matrix construction at line {i}")
    # End of iteration 0
    elif in_iter0 and 'construct_equilibrium_system called, state.iteration=1' in line:
        in_iter0 = False
        break
    # Collect matrix data
    elif in_iter0:
        if 'Writing row for stable phase:' in line:
            # Extract the energy and masses
            for j in range(1, 5):
                if i+j < len(cpu_lines):
                    next_line = cpu_lines[i+j]
                    if 'Energy:' in next_line:
                        energy = re.search(r'Energy:\s*([-\d.e+]+)', next_line)
                        if energy:
                            rhs_values.append(float(energy.group(1)))
                    elif 'Masses:' in next_line:
                        masses = re.findall(r'[-\d.e+]+', next_line.split('Masses:')[1])
                        if masses:
                            matrix_rows.append([float(m) for m in masses[:3]])
                    elif 'Row values' in next_line:
                        values = re.findall(r'[-\d.e+]+', next_line.split('Row values')[1])
                        if values and len(matrix_rows) > len(values):
                            # Update the last row
                            pass

# Print the energy balance rows
print("\nEnergy Balance Rows (First 3 rows):")
for i, (row, rhs) in enumerate(zip(matrix_rows[:3], rhs_values[:3])):
    print(f"  Phase {i}: Masses={row}, Energy RHS={rhs:.2f}")

# Look for constraint rows
print("\nConstraint Rows (Mole fraction constraints):")
constraint_data = []
for i, line in enumerate(cpu_lines):
    if 'write_row_fixed_mole_fraction' in line and 'iteration=0' in cpu_lines[max(0,i-20):i]:
        # Extract phase and component indices
        phase_match = re.search(r'phase_idx=(\d+)', line)
        comp_match = re.search(r'component_idx=(\d+)', line)
        if phase_match and comp_match:
            phase_idx = int(phase_match.group(1))
            comp_idx = int(comp_match.group(1))
            # Get RHS value from next line
            if i+1 < len(cpu_lines) and 'out_rhs' in cpu_lines[i+1]:
                rhs_match = re.search(r'entry:\s*([-\d.e+]+)', cpu_lines[i+1])
                if rhs_match:
                    rhs_val = float(rhs_match.group(1))
                    constraint_data.append((phase_idx, comp_idx, rhs_val))
                    print(f"  Phase {phase_idx}, Component {comp_idx}: RHS={rhs_val:.6f}")

# Try to reconstruct the full matrix
print("\n" + "=" * 80)
print("RECONSTRUCTED CPU MATRIX STRUCTURE")
print("-" * 40)

# Based on the output, we have:
# - 3 phases initially
# - 2 composition constraints (X(CU) and X(FE))
# - Matrix is 6x5: [μ₁, μ₂, NP₁, NP₂, NP₃]

print("\nExpected CPU Matrix (6x5):")
print("Columns: [μ(AL), μ(CU), NP(phase0), NP(phase1), NP(phase2)]")
print()

# Energy balance rows (first 3)
print("Energy balance rows:")
if len(matrix_rows) >= 3 and len(rhs_values) >= 3:
    for i in range(3):
        print(f"Row {i}: {matrix_rows[i]} + [0, 0] for phase amounts | RHS: {rhs_values[i]:.2f}")

# Mass balance rows would be next, but they're more complex

# Constraint rows
print("\nMole fraction constraint rows:")
cu_constraints = [(p, r) for p, c, r in constraint_data if c == 1]  # CU is component 1
fe_constraints = [(p, r) for p, c, r in constraint_data if c == 2]  # FE is component 2

print(f"X(CU) constraint RHS values by phase: {cu_constraints}")
print(f"X(FE) constraint RHS values by phase: {fe_constraints}")

# Try another approach - look for the actual matrix values
print("\n" + "=" * 80)
print("SEARCHING FOR ACTUAL MATRIX VALUES")
print("-" * 40)

# Search for matrix element patterns
matrix_patterns = [
    r'equilibrium_matrix\[.*\]\[.*\]',
    r'A\[.*\]\[.*\]',
    r'matrix.*row.*col',
    r'out_matrix\[.*\]',
    r'eq_matrix.*=',
]

found_matrix = False
for pattern in matrix_patterns:
    matches = []
    for line in cpu_lines:
        if re.search(pattern, line):
            matches.append(line.strip())
    if matches:
        print(f"\nFound {len(matches)} lines matching pattern '{pattern}':")
        for match in matches[:10]:
            print(f"  {match}")
        found_matrix = True

if not found_matrix:
    print("Could not find explicit matrix values in CPU output")
    print("The CPU code may not output the full matrix directly")

# Summary
print("\n" + "=" * 80)
print("CPU MATRIX SUMMARY FOR ITERATION 0")
print("-" * 40)
print("\nBased on the debug output:")
print("1. Matrix dimensions: 6x5")
print("2. 3 phases active initially")
print("3. 2 composition constraints (X(CU) and X(FE))")
print("\n4. RHS values for constraints:")
for p, c, r in constraint_data:
    comp_name = ['AL', 'CU', 'FE'][c] if c < 3 else f'comp{c}'
    print(f"   Phase {p}, X({comp_name}): {r:.6f}")

print("\n5. Key difference from GPU:")
print("   - CPU RHS values are much smaller (order of 0.001-0.03)")
print("   - GPU RHS values are larger (order of 0.1)")
print("   - This leads to larger solution vector in GPU")