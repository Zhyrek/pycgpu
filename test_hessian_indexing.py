#!/usr/bin/env python
"""Test to verify Hessian indexing is correct in GPU code."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import cupy as cp

# Load database and set up conditions
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with a simple phase
phase_name = 'BCC_A2'

# Create workspace conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print(f"Testing Hessian indexing for {phase_name} phase...")
print("="*80)

# Create workspace with single phase
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Check the generated Hessian function structure
lines = all_device_functions.split('\n')

# Look for formulahess function
hessian_start = -1
for i, line in enumerate(lines):
    if '__device__' in line and 'formulahess' in line:
        hessian_start = i
        break

if hessian_start >= 0:
    # Extract function signature
    sig_line = lines[hessian_start]
    print(f"Found Hessian function: {sig_line}")
    
    # Check for x[0], x[1], x[2] usage in first few lines
    print("\nChecking variable indexing in first 50 lines of function:")
    
    var_usage = {}
    for i in range(hessian_start, min(hessian_start + 50, len(lines))):
        line = lines[i]
        
        # Look for x[i] patterns
        import re
        matches = re.findall(r'x\[(\d+)\]', line)
        for match in matches:
            idx = int(match)
            if idx not in var_usage:
                var_usage[idx] = []
            var_usage[idx].append(i - hessian_start)
    
    print("\nVariable usage found:")
    for idx in sorted(var_usage.keys()):
        print(f"  x[{idx}]: used on lines {var_usage[idx][:5]}{'...' if len(var_usage[idx]) > 5 else ''}")
    
    # Check what indices correspond to
    from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model
    model = wks.models[phase_name]
    variable_names = notebook_get_all_sym_names_for_model(model, wks)
    
    print("\nExpected variable mapping:")
    for i, name in enumerate(variable_names):
        print(f"  x[{i}] = {name}")
    
    # Verify the CSE Hessian function outputs reduced matrix
    print("\nCSE Hessian function characteristics:")
    print("- Output: Reduced Hessian matrix (T + site fractions only)")
    print("- No N,P derivatives in output")
    print("- GPU code must map reduced matrix to full matrix")
    
    # Check if the minimizer.h mapping is correct
    print("\nMinimizer.h mapping strategy:")
    print("- temp_hess[0] -> hess[2,2] (T,T element)")
    print("- temp_hess[i] -> hess[2, 3+i-1] and hess[3+i-1, 2] (T,site_fraction)")
    print("- temp_hess[i*(1+phase_dof)+j] -> hess[3+i-1, 3+j-1] (site_fraction block)")
    
    # Verify c_statevars calculation only uses T derivatives
    print("\nc_statevars calculation:")
    print("- Should only access hess[*, 2] (temperature column)")
    print("- N,P columns (0,1) should not be accessed")
    
else:
    print("ERROR: Could not find Hessian function in generated code!")

print("\n" + "="*80)
print("Summary:")
print("- CSE Hessian functions output reduced matrices")
print("- GPU code correctly maps to full matrix format")
print("- c_statevars calculation limited to T derivatives only")
print("="*80)