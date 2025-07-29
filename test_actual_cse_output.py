#!/usr/bin/env python
"""Test actual CSE gradient output by examining generated code."""

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

print("Checking actual CSE gradient function output order...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Find the gradient function
lines = all_device_functions.split('\n')
in_grad_func = False
out_assignments = []

for i, line in enumerate(lines):
    if '__device__' in line and 'formulagrad' in line and 'hess' not in line:
        in_grad_func = True
        print(f"Found gradient function at line {i}:")
        print(f"  {line.strip()}")
        continue
    
    if in_grad_func:
        if line.strip() == '}':
            break
        if 'out[' in line and '=' in line:
            # Extract the output index
            import re
            match = re.search(r'out\[(\d+)\]', line)
            if match:
                idx = int(match.group(1))
                out_assignments.append((idx, line.strip()))

print(f"\nGradient function has {len(out_assignments)} outputs")
print("\nOutput assignments in order:")
for idx, assignment in sorted(out_assignments):
    print(f"  out[{idx}] = {assignment}")

# Try to determine what each output represents
print("\n" + "="*80)
print("Analysis:")
if len(out_assignments) == 3:
    print("CSE gradient outputs 3 values (reduced format)")
    print("Need to determine the order by examining the expressions...")
    
    # Look for clues in the assignments
    for idx, assignment in sorted(out_assignments):
        if 'x[2]' in assignment:  # x[2] is usually T
            print(f"  out[{idx}] likely contains T derivative (has x[2])")
        if 'x[3]' in assignment or 'x[4]' in assignment:  # x[3], x[4] are site fractions
            print(f"  out[{idx}] likely contains site fraction derivatives")

print("="*80)