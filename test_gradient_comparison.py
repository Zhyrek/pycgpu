#!/usr/bin/env python
"""Check if gradient functions also output reduced arrays."""

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

print(f"Checking gradient function output for {phase_name} phase")
print("="*80)

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Get the phase record from CPU
phase_record = wks.phase_record_factory[phase_name]
model = wks.models[phase_name]

# Test point
y_ti = 0.05879468
test_dof = np.array([1.0, 101325.0, 500.0, 1.0 - y_ti, y_ti])

# Calculate CPU gradient
cpu_grad = np.zeros(phase_record.num_statevars + phase_record.phase_dof)
phase_record.formulagrad(cpu_grad, test_dof)

print("CPU gradient:")
for i, val in enumerate(cpu_grad):
    if abs(val) > 1e-10:
        var_name = ['N', 'P', 'T', 'Y_NB', 'Y_TI'][i] if i < 5 else f'var_{i}'
        print(f"  grad[{i}] ({var_name}) = {val:.6e}")

print(f"\nCPU gradient size: {len(cpu_grad)}")
print(f"Expected size: {phase_record.num_statevars + phase_record.phase_dof}")

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Find the gradient function
lines = all_device_functions.split('\n')
grad_found = False
for i, line in enumerate(lines):
    if '__device__' in line and 'formulagrad' in line and 'hess' not in line:
        grad_found = True
        print(f"\n{'='*80}")
        print(f"Found GPU gradient function at line {i}")
        print(f"Function signature: {line.strip()}")
        
        # Count output assignments
        out_count = 0
        for j in range(i, len(lines)):
            if lines[j].strip() == '}':
                break
            if 'out[' in lines[j] and '=' in lines[j]:
                out_count += 1
        
        print(f"Total output assignments: {out_count}")
        
        # Check the first few out[] assignments
        print("\nFirst few output assignments:")
        out_shown = 0
        for j in range(i, len(lines)):
            if lines[j].strip() == '}':
                break
            if 'out[' in lines[j] and '=' in lines[j] and out_shown < 5:
                print(f"  Line {j}: {lines[j].strip()}")
                out_shown += 1
        
        # For BCC_A2 gradient with CSE, expected output:
        # If reduced: out[0]=dG/dT, out[1]=dG/dY_NB, out[2]=dG/dY_TI (size=3)
        # If full: out[0]=dG/dN, out[1]=dG/dP, out[2]=dG/dT, out[3]=dG/dY_NB, out[4]=dG/dY_TI (size=5)
        
        print(f"\nAnalysis:")
        if out_count == 3:
            print("✗ GRADIENT IS REDUCED! Only outputting T and site fraction derivatives.")
            print("  This is the source of the divergence!")
            print("  GPU code expects full gradient array including N,P derivatives.")
        elif out_count == 5:
            print("✓ Gradient includes all derivatives (N, P, T, site fractions)")
        else:
            print(f"? Unexpected gradient size: {out_count}")
        
        break

if not grad_found:
    print("\nERROR: Could not find GPU gradient function!")

print("\n" + "="*80)
print("Summary:")
print("- Hessian functions output reduced matrices (T + site fractions only)")
print("- Need to check if gradient functions are also reduced")
print("- If gradients are reduced, minimizer.h needs to map them like Hessians")
print("="*80)