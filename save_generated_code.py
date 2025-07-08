#!/usr/bin/env python
"""Save the generated C code to examine it."""

from pycalphad import Database, calculate, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import numpy as np

# Load database and set up system
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

# Set up conditions for test
T = 1300.0
conditions = {v.T: T, v.P: 101325, v.X('TI'): 0.5}

# Run calculate to create workspace
calc_result = calculate(db, comps, phases, T=T, P=101325, N=1, output='GM')

# Create workspace
wks = Workspace(db, comps, phases, conditions, calc_result, 
                parameters={}, phase_record_factory=None, verbose=True)

print("Generating C code for phase models...")
model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = \
    _generate_c_code_for_phase_models(wks, include_hess=True, validate=True)

# Save the generated model functions to a file
with open('generated_model_functions.c', 'w') as f:
    f.write(model_funcs_c)
    
print(f"Saved {len(model_funcs_c)} characters of C code to generated_model_functions.c")

# Also save init calls
with open('generated_init_calls.c', 'w') as f:
    for call in pr_init_calls_c:
        f.write(call + '\n')
        
print(f"Saved {len(pr_init_calls_c)} init calls to generated_init_calls.c")

# Check the first few lines of each model's functions
import re
model_0_funcs = re.findall(r'__device__.*?pycgpu_model_0_\w+.*?\{', model_funcs_c, re.DOTALL)
model_1_funcs = re.findall(r'__device__.*?pycgpu_model_1_\w+.*?\{', model_funcs_c, re.DOTALL)

print(f"\nModel 0 function signatures found: {len(model_0_funcs)}")
for sig in model_0_funcs[:3]:
    print(f"  {sig.split('{')[0].strip()}")
    
print(f"\nModel 1 function signatures found: {len(model_1_funcs)}")  
for sig in model_1_funcs[:3]:
    print(f"  {sig.split('{')[0].strip()}")