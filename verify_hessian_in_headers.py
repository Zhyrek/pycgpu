#!/usr/bin/env python3
"""Verify if Hessian fix is applied in generated code"""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.model import Model
from pycalphad.gpu.gpu_codegen import notebook_source_from_expr
import pycalphad.variables as v

# Load database and create workspace
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create models
models = {}
for phase in phases:
    models[phase] = Model(db, comps, phase)

# Create workspace
state_vars = [v.T]
conditions = {v.T: 300, v.P: 101325}
wks = Workspace(db, comps, phases, conditions, models=models, phase_record_factory=None)

# Generate Hessian with verbose output
print("Generating Hessian with verbose=True to see fix messages...")
model = models['BCC_A2']
hess_code = notebook_source_from_expr(
    model.G, 
    "formulahess", 
    model, 
    0, 
    wks, 
    expr_type="hess", 
    c_output_type="void", 
    validate=True, 
    verbose=True
)

# Check if the spurious terms are present
print("\nChecking for spurious terms in generated code...")
print("Looking for pow(x[4], (-1)) in diagonal element for x[3]...")

# Find out[18] which should be the diagonal element
import re
out18_match = re.search(r'out\[18\] = ([^;]+);', hess_code)
if out18_match:
    out18_expr = out18_match.group(1)
    print(f"\nFound out[18] expression (length={len(out18_expr)}):")
    # Check for spurious terms
    if 'pow(x[4], (-1))' in out18_expr:
        print("ERROR: Spurious term pow(x[4], (-1)) found in diagonal element!")
        # Count occurrences
        count = out18_expr.count('pow(x[4], (-1))')
        print(f"Found {count} occurrences of pow(x[4], (-1))")
    else:
        print("SUCCESS: No spurious pow(x[4], (-1)) terms found!")
        
# Also check out[24] for the other diagonal
out24_match = re.search(r'out\[24\] = ([^;]+);', hess_code)
if out24_match:
    out24_expr = out24_match.group(1)
    print(f"\nFound out[24] expression (length={len(out24_expr)}):")
    if 'pow(x[3], (-1))' in out24_expr:
        print("ERROR: Spurious term pow(x[3], (-1)) found in diagonal element!")
        count = out24_expr.count('pow(x[3], (-1))')
        print(f"Found {count} occurrences of pow(x[3], (-1))")
    else:
        print("SUCCESS: No spurious pow(x[3], (-1)) terms found!")