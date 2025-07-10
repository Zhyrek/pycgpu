#!/usr/bin/env python3
"""Extract and analyze GPU hessian element [3,3] (d²G/dY_NB²)"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model
import re

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory with typical conditions
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

# Create workspace
class MinimalWorkspace:
    def __init__(self, prf):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = prf
        
wks = MinimalWorkspace(prf)

print("=== Generating GPU Hessian Code ===")
hess_code = _nb_formulahess_from_model(model, 0, wks, validate=False, verbose=False)

# Extract the hessian function
lines = hess_code.split('\n')
print(f"\nTotal lines in hessian code: {len(lines)}")

# Find the function definition
for i, line in enumerate(lines):
    if '__device__ void pycgpu_model_0_formulahess' in line:
        print(f"\nFunction starts at line {i}: {line}")
        break

# Element [3,3] in a 5x5 matrix (flattened) is at index 3*5 + 3 = 18
print("\n=== Hessian Element [3,3] (d²G/dY_NB²) ===")
for i, line in enumerate(lines):
    if 'out[18] =' in line:
        print(f"Line {i}: {line.strip()}")
        
        # Parse the expression to understand it better
        expr = line.split('=', 1)[1].strip().rstrip(';')
        print(f"\nExpression: {expr}")
        
        # Check for key patterns
        if '(x[3] + x[4])' in expr:
            print("  -> Contains (x[3] + x[4]) = (Y_NB + Y_TI)")
            
            # Count how many times it appears
            count = expr.count('(x[3] + x[4])')
            print(f"  -> (x[3] + x[4]) appears {count} times")
            
            # Check if it's in denominator
            if '/(x[3] + x[4])' in expr:
                print("  -> (x[3] + x[4]) appears in DENOMINATOR")
                
        # Look for temperature terms
        temp_terms = re.findall(r'x\[2\][^)]*', expr)
        if temp_terms:
            print(f"  -> Temperature terms found: {len(temp_terms)} occurrences")
            
        # Look for large constants
        large_consts = re.findall(r'\d+\.\d+', expr)
        if large_consts:
            print(f"  -> Constants: {', '.join(large_consts[:5])}...")
            
        break

# Also check element [3,4] (d²G/dY_NB∂Y_TI) at index 3*5 + 4 = 19
print("\n=== Hessian Element [3,4] (d²G/dY_NB∂Y_TI) ===")
for i, line in enumerate(lines):
    if 'out[19] =' in line:
        print(f"Line {i}: {line.strip()}")
        expr = line.split('=', 1)[1].strip().rstrip(';')
        if '(x[3] + x[4])' in expr:
            count = expr.count('(x[3] + x[4])')
            print(f"  -> (x[3] + x[4]) appears {count} times")
        break

# Write the full hessian code to a file for inspection
with open('generated_gpu_hessian.c', 'w') as f:
    f.write(hess_code)
print("\n=== Full hessian code written to generated_gpu_hessian.c ===")