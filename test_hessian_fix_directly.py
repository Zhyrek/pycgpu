#!/usr/bin/env python3
"""Test the hessian fix directly"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
from pycalphad.gpu.gpu_codegen import notebook_source_from_expr, _nb_formulahess_from_model

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create minimal workspace
class MinimalWorkspace:
    def __init__(self):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = None
        self.verbose = True
        
wks = MinimalWorkspace()

print("=== Testing Hessian Generation ===")

# Generate hessian
hess_code = _nb_formulahess_from_model(model, 0, wks, validate=False, verbose=True)

# Check for spurious terms (with model-level indices)
if '8.3145*x[0]*(1.0*((1e-15 < x[2]) ? (pow(x[2], (-1))) : 0) + 1.0*((1e-15 < x[1]) ? (pow(x[1], (-1))) : 0))/(x[1] + x[2])' in hess_code:
    print("\nERROR: Spurious term still present in generated code!")
    print("The fix is not being applied correctly.")
else:
    print("\nGOOD: Spurious term not found in generated code.")
    
# Check what the hessian[1,1] element looks like (for model-level, this is index 3)
import re
out3_match = re.search(r'out\[3\] = ([^;]+);', hess_code)
if out3_match:
    out3_expr = out3_match.group(1)
    print(f"\nGenerated hessian[1,1] (first 200 chars):")
    print(out3_expr[:200] + "...")
    
    # Check if it contains the pattern we're trying to fix
    if '8.3145*x[0]*' in out3_expr and 'pow(x[2], (-1))' in out3_expr:
        print("\nWARNING: hessian[1,1] contains RT/Y_TI term (spurious)")
    else:
        print("\nGOOD: hessian[1,1] does not contain RT/Y_TI term")