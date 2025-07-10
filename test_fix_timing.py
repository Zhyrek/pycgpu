#!/usr/bin/env python3
"""Test what the expression looks like when fix is called"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import notebook_replace_piecewise, notebook_replace_exp, fix_ternary_operator_precedence
import pycalphad.variables as v

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

# Convert to string and apply the same transformations as in the code
s = str(d2G_dYNB2)
print("Original expression (first 500 chars):")
print(s[:500] + "...")
print()

# Apply transformations in the same order
s = notebook_replace_piecewise(s)
print("After notebook_replace_piecewise (first 500 chars):")
print(s[:500] + "...")
print()

# Check if we have x[i] format
import re
x_pattern = r'x\[\d+\]'
x_matches = re.findall(x_pattern, s)
print(f"Found {len(x_matches)} occurrences of x[i] pattern")

# Check for spurious terms
spurious_pattern = r'1\.0\*\(\(1e-15 < x\[2\]\) \? \(pow\(x\[2\], \(-1\)\)\) : 0\)'
spurious_matches = re.findall(spurious_pattern, s)
print(f"Found {len(spurious_matches)} occurrences of spurious pattern for x[2]")