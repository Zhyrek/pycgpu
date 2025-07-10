#!/usr/bin/env python3
"""Find the actual entropy terms in the Hessian"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import (notebook_replace_piecewise, notebook_replace_exp, 
                                       fix_ternary_operator_precedence)
import pycalphad.variables as v
import re

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Create model
model = Model(db, comps, 'BCC_A2')

# Get the Hessian expression for Y_NB, Y_NB
Y_NB = model.variables[1]  # Y(BCC_A2,0,NB)
Y_TI = model.variables[2]  # Y(BCC_A2,0,TI)

# Calculate second derivative
dG_dYNB = model.G.diff(Y_NB)
d2G_dYNB2 = dG_dYNB.diff(Y_NB)

# Convert to string and apply transformations
s = str(d2G_dYNB2)
s = notebook_replace_piecewise(s)
s = notebook_replace_exp(s)  
s = fix_ternary_operator_precedence(s)

# Look for different patterns that might be the entropy terms
print("=== Looking for entropy-related terms ===\n")

# Pattern 1: Direct log terms
log_pattern = r'log\(BCC_A20TI\)'
log_matches = re.findall(log_pattern, s)
print(f"Found {len(log_matches)} log(BCC_A20TI) terms")

# Pattern 2: 1/Y terms
inv_pattern = r'pow\(BCC_A20TI, \(-1\.0\)\)'
inv_matches = re.findall(inv_pattern, s)
print(f"Found {len(inv_matches)} pow(BCC_A20TI, (-1.0)) terms")

# Pattern 3: Piecewise with 1/Y
piecewise_inv_pattern = r'\(\(.*?\) \? \(.*?pow\(BCC_A20TI, \(-1\.0\)\).*?\) : \(0\)\)'
piecewise_matches = re.findall(piecewise_inv_pattern, s)
print(f"Found {len(piecewise_matches)} Piecewise 1/Y terms")

# Pattern 4: Look for 8.3145 (R constant) near TI terms
r_constant_pattern = r'8\.3145[^)]*?BCC_A20TI[^)]*?\)'
r_matches = re.findall(r_constant_pattern, s)
print(f"\nFound {len(r_matches)} terms with R constant and BCC_A20TI")

# Pattern 5: Look for any division by BCC_A20TI
div_pattern = r'/BCC_A20TI'
div_matches = re.findall(div_pattern, s)
print(f"Found {len(div_matches)} divisions by BCC_A20TI")

# Look for the specific terms that involve 1/Y
print("\n=== Examining specific 1/Y terms ===")
# Find all occurrences with context
for match in re.finditer(r'.{50}pow\(BCC_A20TI, \(-1\.0\)\).{50}', s):
    print(f"\nContext: ...{match.group()}...")

# Check if we have the Piecewise construct for log(Y)
piecewise_log = r'\(\(1e-15 < BCC_A20TI\)'
if piecewise_log in s:
    print("\n\nFound Piecewise construct for small Y values!")
    # Find the full pattern
    full_pattern = r'\(\(1e-15 < BCC_A20TI\) \? \([^)]+\) : \(0\)\)'
    full_matches = re.findall(full_pattern, s)
    print(f"Found {len(full_matches)} Piecewise constructs")
    for i, match in enumerate(full_matches[:3]):  # Show first 3
        print(f"\nMatch {i+1}: {match}")