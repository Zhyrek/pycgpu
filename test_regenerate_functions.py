#!/usr/bin/env python3
"""Test script to regenerate GPU functions and check variable mapping"""

import sys
import os
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models, notebook_get_all_sym_names_for_model
from pycalphad.model import Model

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing GPU Function Generation ===")

# Create workspace
wks = Workspace(db, comps, phases, eq_conditions)
print("Workspace created")

# Check model details
model = wks.models['BCC_A2']
print(f"Model state variables: {[str(var) for var in model.state_variables]}")
print(f"Model site fractions: {[str(sf) for sf in model.site_fractions]}")

# Check variable mapping
var_names = notebook_get_all_sym_names_for_model(model, wks)
print(f"GPU variable mapping: {dict(zip(var_names, [f'x[{i}]' for i in range(len(var_names))]))}")

# Generate C code
print("\n=== Generating C Code ===")
try:
    model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = \
        _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    print("C code generated successfully")
    print(f"Number of unique models: {len(unique_py_models)}")
    
    # Check if the obj function has the correct variable indexing
    print("\n=== Checking Generated Functions ===")
    print(f"C code length: {len(model_funcs_c)} characters")
    
    # model_funcs_c is a single string containing all functions
    if 'pycgpu_model_0_obj' in model_funcs_c:
        print("Found obj function for model 0")
        
        # Find the obj function
        lines = model_funcs_c.split('\n')
        for i, line in enumerate(lines):
            if 'pycgpu_model_0_obj' in line and '(' in line:
                print(f"Function signature: {line}")
                # Get the next line with the return statement
                if i + 1 < len(lines):
                    return_line = lines[i + 1]
                    print(f"Return statement: {return_line}")
                    
                    # Check if the division is correct
                    if '/(x[3] + x[4])' in return_line:
                        print("✓ CORRECT: Division uses x[3] + x[4] (site fractions)")
                    elif '/(x[1] + x[2])' in return_line:
                        print("✗ INCORRECT: Division uses x[1] + x[2] (P + T)")
                    else:
                        print("? UNCLEAR: Division pattern not found")
                        print(f"  Looking for division patterns in: {return_line}")
                break
    else:
        print("No obj function found in generated code")
    
except Exception as e:
    print(f"Error generating C code: {e}")
    import traceback
    traceback.print_exc()