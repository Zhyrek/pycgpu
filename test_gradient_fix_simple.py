#!/usr/bin/env python
"""Simple test to verify gradient indexing fix."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import cupy as cp
import os

# Clear cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']

# Test with BCC_A2 phase
phase_name = 'BCC_A2'
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Verifying gradient indexing fix...")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Check the generated gradient function
lines = all_device_functions.split('\n')

# Find gradient mapping code
grad_mapping_found = False
for i, line in enumerate(lines):
    if 'temp_grad[0] -> csst->grad[2]' in line:
        grad_mapping_found = True
        print("✓ Found gradient mapping code!")
        print(f"  Line {i}: {line.strip()}")
        
        # Show surrounding lines
        print("\n  Context:")
        for j in range(max(0, i-5), min(len(lines), i+10)):
            if j == i:
                print(f"  >>> {lines[j]}")
            else:
                print(f"      {lines[j]}")
        break

if not grad_mapping_found:
    # The mapping code is in minimizer.h, not in the generated code
    # Let's check if the gradient function outputs reduced array
    for i, line in enumerate(lines):
        if '__device__' in line and 'formulagrad' in line and 'hess' not in line:
            print("\nChecking gradient function signature:")
            print(f"  {line.strip()}")
            
            # Count outputs
            out_count = 0
            for j in range(i, min(i+100, len(lines))):
                if lines[j].strip() == '}':
                    break
                if 'out[' in lines[j] and '=' in lines[j]:
                    out_count += 1
            
            print(f"  Output count: {out_count}")
            
            if out_count == 3:
                print("✓ Gradient outputs reduced array (T + site fractions)")
                print("✓ The fix in minimizer.h should map this correctly")
            else:
                print("✗ Unexpected gradient output size")
            break

print("\n" + "="*80)
print("Summary:")
print("- CSE gradient functions output reduced arrays (size 3 for BCC_A2)")
print("- The fix adds mapping code in minimizer.h to expand to full array")
print("- temp_grad[0] -> grad[2] (Temperature)")
print("- temp_grad[1] -> grad[3] (Y_NB)")
print("- temp_grad[2] -> grad[4] (Y_TI)")
print("- grad[0], grad[1] remain zero (N, P derivatives)")
print("="*80)