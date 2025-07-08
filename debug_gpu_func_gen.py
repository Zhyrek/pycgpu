#!/usr/bin/env python
"""Debug GPU function generation"""

import numpy as np
from pycalphad import Database, Workspace
import pycalphad.variables as v
from pycalphad.gpu.gpu_codegen import notebook_source_from_expr

# Simple binary system for testing
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1000, v.P: 101325, v.X('TI'): 0.4}

# Create workspace
wks = Workspace(database=dbf, components=comps, phases=phases, 
                conditions=conds, models=None, parameters=None,
                calc_opts={'pdens': 10}, verbose=False)

model = wks.models['BCC_A2']

print("=== GPU FUNCTION GENERATION DEBUG ===")

# Generate the obj function
obj_func_code = notebook_source_from_expr(
    model.GM, "obj", model, 0, wks, 
    expr_type="func", c_output_type="double", 
    validate=False, verbose=True
)

print("\n=== GENERATED OBJ FUNCTION ===")
print(obj_func_code)  # Full function
print(f"\nFunction length: {len(obj_func_code)} characters")

# Check variable conversion
from pycalphad.gpu.gpu_codegen import notebook_get_all_sym_names_for_model, notebook_convert_var_names

var_names = notebook_get_all_sym_names_for_model(model, wks)
print(f"\nVariable names in order: {var_names}")

# Test conversion of a simple expression
test_expr = "N + P + T + BCC_A20NB + BCC_A20TI"
converted = notebook_convert_var_names(test_expr, model, wks)
print(f"\nTest expression: {test_expr}")
print(f"Converted: {converted}")

# Check a more complex expression with logs
test_expr2 = "BCC_A20NB*log(BCC_A20NB) + BCC_A20TI*log(BCC_A20TI)"
converted2 = notebook_convert_var_names(test_expr2, model, wks)
print(f"\nLog expression: {test_expr2}")
print(f"Converted: {converted2}")