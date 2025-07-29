#!/usr/bin/env python
"""Inspect the generated gradient function."""

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

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Write to file for inspection
with open('generated_gradient_function.c', 'w') as f:
    f.write(all_device_functions)

print("Generated gradient function written to generated_gradient_function.c")
print("\nSearching for gradient function...")

# Find and print gradient function
lines = all_device_functions.split('\n')
in_grad = False
grad_lines = []

for line in lines:
    if 'formulagrad' in line and '__device__' in line and 'hess' not in line:
        in_grad = True
    if in_grad:
        grad_lines.append(line)
        if line.strip() == '}' and in_grad:
            break

print(f"\nFound gradient function with {len(grad_lines)} lines")
print("\nFirst 50 lines of gradient function:")
for line in grad_lines[:50]:
    print(line)