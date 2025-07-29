#!/usr/bin/env python
"""Test actual CSE gradient output by generating and inspecting the code."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phase_name = 'BCC_A2'

conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Inspecting actual CSE gradient function output...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Find the gradient function
lines = all_device_functions.split('\n')
in_grad_func = False
grad_func_lines = []

for i, line in enumerate(lines):
    if '__device__' in line and 'formulagrad' in line and 'hess' not in line:
        in_grad_func = True
        print(f"Found gradient function:")
        print(f"  {line.strip()}")
        continue
    
    if in_grad_func:
        grad_func_lines.append(line)
        if line.strip() == '}':
            break

# Analyze the output assignments
print("\nGradient function body:")
out_assignments = []
for line in grad_func_lines[:20]:  # First 20 lines
    print(line)
    if 'out[' in line and '=' in line:
        # Extract the output index
        import re
        match = re.search(r'out\[(\\d+)\\]', line)
        if match:
            idx = int(match.group(1))
            out_assignments.append((idx, line.strip()))

print(f"\nFound {len(out_assignments)} output assignments")
print("\nFirst few assignments:")
for idx, assignment in out_assignments[:5]:
    print(f"  out[{idx}]: {assignment[:100]}...")

# Try to determine what each output represents by looking for telltale signs
print("\nAnalyzing output order:")
for idx, assignment in out_assignments:
    if 'x[2]' in assignment and 'x[3]' not in assignment and 'x[4]' not in assignment:
        print(f"  out[{idx}] appears to be pure temperature derivative (contains x[2] only)")
    elif 'log' in assignment or 'x[3]' in assignment or 'x[4]' in assignment:
        if 'x[3]' in assignment and 'x[4]' not in assignment:
            print(f"  out[{idx}] likely dG/dY(NB) (contains x[3])")
        elif 'x[4]' in assignment and 'x[3]' not in assignment:
            print(f"  out[{idx}] likely dG/dY(TI) (contains x[4])")
        else:
            print(f"  out[{idx}] contains both site fractions")

print("="*80)