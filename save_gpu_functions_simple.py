#!/usr/bin/env python3
"""Save generated GPU functions to file for inspection"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v

# Load database and create models
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Create workspace
wks = Workspace(db, comps, phases, {v.N: 1, v.P: 101325, v.T: 1000}, verbose=False)

# Generate C code for phase models
print("Generating C code for phase models...")
all_model_device_functions_c_code, g_phase_record_array_init_calls_c_code, unique_py_models, py_phase_name_to_unique_idx_map = _generate_c_code_for_phase_models(wks, include_hess=True)

# Save to file
with open('gpu_generated_functions.cu', 'w') as f:
    f.write(all_model_device_functions_c_code)
    f.write("\n// Phase record initialization calls:\n")
    for call in g_phase_record_array_init_calls_c_code:
        f.write(call)

print("Generated GPU code saved to gpu_generated_functions.cu")
print(f"Total length: {len(all_model_device_functions_c_code)} characters")
print(f"Phase mapping: {py_phase_name_to_unique_idx_map}")