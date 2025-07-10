#!/usr/bin/env python3
"""Test what expressions look like at each stage"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v

# Create model
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 1800, v.P: 101325, v.X('TI'): 0.3}

wks = Workspace(database=db, components=comps, phases=phases, conditions=conds)
model = wks.models['BCC_A2']

# Get variables
Y_NB = model.variables[1]  # Y(BCC_A2,0,NB)
Y_TI = model.variables[2]  # Y(BCC_A2,0,TI)

# Calculate Hessian element [3,3] (d²G/dY_NB²)
dG_dYNB = model.G.diff(Y_NB)
d2G_dYNB2 = dG_dYNB.diff(Y_NB)

print("=== Original Hessian expression ===")
expr_str = str(d2G_dYNB2)
print(f"Length: {len(expr_str)}")
print(f"Contains log(BCC_A20TI): {'log(BCC_A20TI)' in expr_str}")
print(f"Contains BCC_A20TI**(-1): {'BCC_A20TI**(-1)' in expr_str}")

# Apply the transformations
from pycalphad.gpu.gpu_codegen import (notebook_replace_piecewise, notebook_replace_exp,
                                       fix_ternary_operator_precedence, notebook_convert_var_names)

# Step 1: Replace Piecewise
expr_str = notebook_replace_piecewise(expr_str)
print("\n=== After Piecewise replacement ===")
print(f"Contains pow(BCC_A20TI, (-1)): {'pow(BCC_A20TI, (-1))' in expr_str}")

# Step 2: Convert variable names
expr_str = notebook_convert_var_names(expr_str, model, wks)
print("\n=== After variable conversion ===")
print(f"Contains pow(x[4], (-1)): {'pow(x[4], (-1))' in expr_str}")

# Count occurrences
import re
x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', expr_str))
print(f"Number of pow(x[4], (-1)) terms: {x4_count}")

# Check if the exact pattern is there
test_pattern = r'1\.0\*\(\(1e-15 < x\[4\]\) \? \(pow\(x\[4\], \(-1\)\)\) : 0\)'
matches = re.findall(test_pattern, expr_str)
print(f"Exact pattern matches: {len(matches)}")