#!/usr/bin/env python
"""Check variable mappings being generated for the LIQUID phase."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import os
import time

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Create workspace conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Checking variable mappings for LIQUID phase...")
print("="*80)

# Process LIQUID phase specifically
phase_name = 'LIQUID'
print(f"\nProcessing {phase_name}...")

try:
    # Create workspace with single phase
    wks = Workspace(db, components, [phase_name], conditions, verbose=False)
    
    # Access the model to get variable information
    model = wks.models[phase_name]
    
    print(f"Phase: {phase_name}")
    print(f"Model state variables: {[str(var) for var in model.state_variables]}")
    
    if hasattr(wks, 'phase_record_factory') and wks.phase_record_factory is not None:
        print(f"Workspace state variables: {[str(var) for var in wks.phase_record_factory.state_variables]}")
    else:
        print("No phase_record_factory found")
    
    print(f"Model site fractions: {[str(sf) for sf in model.site_fractions]}")
    
    # Get the actual variable mapping that will be used
    from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model
    variable_names = notebook_get_all_sym_names_for_model(model, wks)
    
    print(f"\nActual variable mapping:")
    for i, name in enumerate(variable_names):
        print(f"  x[{i}] = {name}")
    
    # Generate code to trigger the debug output
    print(f"\nGenerating C code to see actual mappings...")
    result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    all_device_functions, init_calls, unique_models, phase_map = result
    
    # Find and extract the Hessian function to see the actual x[i] usage
    lines = all_device_functions.split('\n')
    
    # Look for formulahess function
    hessian_start = -1
    hessian_end = -1
    for i, line in enumerate(lines):
        if '__device__' in line and 'formulahess' in line:
            hessian_start = i
        elif hessian_start >= 0 and line.strip() == '}':
            hessian_end = i
            break
    
    if hessian_start >= 0 and hessian_end >= 0:
        # Extract first few lines of Hessian function to see variable usage
        hessian_lines = lines[hessian_start:min(hessian_end+1, hessian_start+20)]
        
        print(f"\nFirst 20 lines of Hessian function:")
        for i, line in enumerate(hessian_lines):
            print(f"  {i:2d}: {line}")
        
        # Look for x[i] patterns in the function
        import re
        variable_usage = {}
        full_hessian = '\n'.join(lines[hessian_start:hessian_end+1])
        
        matches = re.findall(r'x\[(\d+)\]', full_hessian)
        for match in matches:
            idx = int(match)
            if idx not in variable_usage:
                variable_usage[idx] = 0
            variable_usage[idx] += 1
        
        print(f"\nVariable usage in Hessian function:")
        for idx in sorted(variable_usage.keys()):
            var_name = variable_names[idx] if idx < len(variable_names) else "UNKNOWN"
            print(f"  x[{idx}] ({var_name}): used {variable_usage[idx]} times")
            
    else:
        print(f"Could not find Hessian function")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")

print(f"\n{'='*80}")
print("Summary:")
print("The expected pycalphad pattern should be:")
print("  x[0] = N (moles)")
print("  x[1] = P (pressure)")  
print("  x[2] = T (temperature)")
print("  x[3] = X_AL (mole fraction)")
print("  x[4] = X_CU (mole fraction)")
print("  x[5] = X_FE (mole fraction)")
print("="*80)