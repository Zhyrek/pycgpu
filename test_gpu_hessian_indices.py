#!/usr/bin/env python3
"""Test if GPU hessian function has correct variable indices"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, Workspace, calculate
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model, notebook_get_all_syms_for_model

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create workspace to get proper state variables
from pycalphad.core.workspace import Workspace
wks = Workspace(db, ['NB', 'TI', 'VA'], ['BCC_A2'], conditions={'T': 1000, 'P': 101325, 'N': 1})

# Actually, let's just create a minimal workspace object
class MinimalWorkspace:
    def __init__(self):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = None
        
wks = MinimalWorkspace()

print("=== Variable Ordering Analysis ===")
print(f"Model state variables: {model.state_variables}")
print(f"Model site fractions: {model.site_fractions}")

# Check what notebook_get_all_syms_for_model returns
all_syms = notebook_get_all_syms_for_model(model, wks)
print(f"\nGPU variable ordering: {all_syms}")
print(f"Total GPU variables: {len(all_syms)}")

# Check workspace state variables
if hasattr(wks, 'phase_record_factory') and wks.phase_record_factory is not None:
    print(f"\nWorkspace state variables: {wks.phase_record_factory.state_variables}")
else:
    print("\nNo phase_record_factory in workspace")

# Generate hessian code to inspect
print("\n=== Generating GPU Hessian Code ===")
hess_code = _nb_formulahess_from_model(model, 0, wks, validate=False, verbose=True)

# Extract the variable mapping from the generated code
print("\n=== Variable Index Mapping ===")
import re
# Look for x[0], x[1], etc. in the code
indices_used = set()
for match in re.finditer(r'x\[(\d+)\]', hess_code):
    indices_used.add(int(match.group(1)))

print(f"Indices used in hessian: {sorted(indices_used)}")

# Now let's check what the hessian expects
print("\n=== Expected Variable Values ===")
print("If GPU expects workspace DOF [N, P, T, Y_NB, Y_TI]:")
print("  x[0] = N = 1.0")
print("  x[1] = P = 101325.0") 
print("  x[2] = T = 1000.0")
print("  x[3] = Y_NB = 0.6")
print("  x[4] = Y_TI = 0.4")

print("\n=== Testing Hessian Element [3,3] ===")
# Extract just the [3,3] element calculation
# In flattened format with 5 variables, element [3,3] is at index 3*5 + 3 = 18
lines = hess_code.split('\n')
for i, line in enumerate(lines):
    if 'out[18]' in line:
        print(f"Hessian [3,3] calculation:")
        print(f"  {line.strip()}")
        # Check if it uses the site fraction sum
        if 'x[3] + x[4]' in line:
            print("  -> Uses (x[3] + x[4]) which should be (Y_NB + Y_TI) = 1.0")