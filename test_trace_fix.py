#!/usr/bin/env python
"""Trace the fix_missing_operators application"""

from pycalphad import Database, Model
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models

db = Database('Al-Cu-Fe.tdb')

# Test ALCU_ZETA phase
phase_name = 'ALCU_ZETA'
print(f"\nTracing code generation for {phase_name}...")

models = {}
for phase in [phase_name]:
    mod = Model(db, ['AL', 'CU', 'FE'], phase)
    models[phase] = mod

# Get generated code
from pycalphad.core.workspace import Workspace
wks = Workspace(database=db, components=['AL', 'CU', 'FE'], phases=[phase_name], 
                conditions={}, models=models, verbose=True)

# Generate code
code_tuple = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
code = code_tuple[0] if isinstance(code_tuple, tuple) else code_tuple

# Check if the pattern exists
import re
matches = re.findall(r'1 11\.0\*', code)
print(f"\nFound {len(matches)} occurrences of '1 11.0*' pattern in generated code")

# Save it
with open('test_generated_code.c', 'w') as f:
    f.write(code)
    
print("Code saved to test_generated_code.c")