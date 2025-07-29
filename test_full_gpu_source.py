#!/usr/bin/env python
"""Test full GPU source generation with fix"""

import sys
sys.path.insert(0, '/mnt/c/users/scott/Documents/pycalphad')

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models, _generate_full_gpu_source
import re

# Setup
db = Database('Al-Cu-Fe.tdb')
phase_name = 'ALCU_ZETA'

models = {}
for phase in [phase_name]:
    mod = Model(db, ['AL', 'CU', 'FE'], phase)
    models[phase] = mod

# Create workspace
wks = Workspace(database=db, components=['AL', 'CU', 'FE'], phases=[phase_name], 
                conditions={}, models=models, verbose=True)

# Generate code
code_tuple = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
model_funcs_c = code_tuple[0]
pr_init_calls_c = code_tuple[1]  
unique_py_models = code_tuple[2]
num_unique_models_for_gpu = len(unique_py_models)

# Generate full GPU source
full_kernel_source = _generate_full_gpu_source(wks, model_funcs_c, pr_init_calls_c, num_unique_models_for_gpu)

# Check if the pattern exists
matches = re.findall(r'1 11\.0\*', full_kernel_source)
print(f"\nFound {len(matches)} occurrences of '1 11.0*' pattern in full GPU source")

if len(matches) == 0:
    print("SUCCESS: All patterns have been fixed!")
else:
    print("FAILED: Pattern still exists")
    # Show a few examples
    for i, match in enumerate(matches[:3]):
        # Find context around the match
        idx = full_kernel_source.find(match)
        context_start = max(0, idx - 50)
        context_end = min(len(full_kernel_source), idx + 50)
        context = full_kernel_source[context_start:context_end]
        print(f"\nExample {i+1} context: ...{context}...")

# Save it
with open('test_full_gpu_source.cu', 'w') as f:
    f.write(full_kernel_source)
    
print("\nFull GPU source saved to test_full_gpu_source.cu")