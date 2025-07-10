#!/usr/bin/env python3
"""Trace the differentiation process to see where 1/Y terms come from"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import (notebook_replace_piecewise, notebook_replace_exp, 
                                       fix_ternary_operator_precedence)
import pycalphad.variables as v
import symengine as se

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Create model
model = Model(db, comps, 'BCC_A2')

# Get the variables
Y_NB = model.variables[1]  # Y(BCC_A2,0,NB)
Y_TI = model.variables[2]  # Y(BCC_A2,0,TI)

print("=== Checking model.G for log terms ===")
G_str = str(model.G)
print(f"log(Y(BCC_A2,0,NB)) count in G: {G_str.count('log(Y(BCC_A2,0,NB))')}")
print(f"log(Y(BCC_A2,0,TI)) count in G: {G_str.count('log(Y(BCC_A2,0,TI))')}")

# Look at first derivative
print("\n=== First derivative dG/dY_NB ===")
dG_dYNB = model.G.diff(Y_NB)
dG_str = str(dG_dYNB)

# Check for 1/Y terms in first derivative
print(f"Contains 1/Y(BCC_A2,0,NB): {'Y(BCC_A2,0,NB)**(-1)' in dG_str}")
print(f"Contains 1/Y(BCC_A2,0,TI): {'Y(BCC_A2,0,TI)**(-1)' in dG_str}")

# Now check second derivative
print("\n=== Second derivative d²G/dY_NB² ===")
d2G_dYNB2 = dG_dYNB.diff(Y_NB)
d2G_str = str(d2G_dYNB2)

# Check for 1/Y² terms
print(f"Contains 1/Y(BCC_A2,0,NB)²: {'Y(BCC_A2,0,NB)**(-2)' in d2G_str}")
print(f"Contains 1/Y(BCC_A2,0,TI)²: {'Y(BCC_A2,0,TI)**(-2)' in d2G_str}")

# Check what Piecewise does to log terms
print("\n=== Checking Piecewise handling of log ===")
# The model likely uses Piecewise to handle log(0) issues
if 'Piecewise' in G_str:
    print("Model.G contains Piecewise constructs")
    # Extract a Piecewise log term
    import re
    piecewise_log_pattern = r'Piecewise\([^)]*log\([^)]+\)[^)]*\)'
    matches = re.findall(piecewise_log_pattern, G_str)
    if matches:
        print(f"Found {len(matches)} Piecewise log constructs")
        print(f"Example: {matches[0][:100]}...")

# Now let's see what notebook_replace_piecewise does
print("\n=== After notebook_replace_piecewise ===")
d2G_replaced = notebook_replace_piecewise(d2G_str)

# Check if we now have pow(Y, -1)
if 'pow(Y(BCC_A2,0,TI), (-1))' in d2G_replaced:
    print("Found pow(Y(BCC_A2,0,TI), (-1)) after Piecewise replacement!")
elif 'pow(Y(BCC_A2,0,TI), (-1.0))' in d2G_replaced:
    print("Found pow(Y(BCC_A2,0,TI), (-1.0)) after Piecewise replacement!")
else:
    print("No direct pow(Y, -1) terms found")
    
# Check if the issue is with variable name conversion
print("\n=== Checking variable name format ===")
# In symengine, it might be BCC_A20TI instead of Y(BCC_A2,0,TI)
if 'BCC_A20TI**(-1)' in d2G_str:
    print("Found BCC_A20TI**(-1) in second derivative!")
elif 'BCC_A20TI**(-2)' in d2G_str:
    print("Found BCC_A20TI**(-2) in second derivative!")
    
# Let's look for the actual pattern after all transformations
d2G_final = fix_ternary_operator_precedence(notebook_replace_exp(d2G_replaced))
print("\n=== Final transformed expression check ===")
print(f"Contains 'BCC_A20TI**(-1)': {'BCC_A20TI**(-1)' in d2G_final}")
print(f"Contains 'pow(BCC_A20TI, (-1))': {'pow(BCC_A20TI, (-1))' in d2G_final}")
print(f"Contains 'pow(BCC_A20TI, (-1.0))': {'pow(BCC_A20TI, (-1.0))' in d2G_final}")