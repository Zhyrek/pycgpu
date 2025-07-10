#!/usr/bin/env python3
"""Debug why fix_hessian_spurious_terms_v2 isn't working"""

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

print("Looking for entropy terms in Hessian before variable conversion...")
print(f"Expression length: {len(s)}")

# Look for the actual pattern in the expression
# The 1/Y terms come from log(Y) differentiation
# d/dY log(Y) = 1/Y
# d²/dY² log(Y) = -1/Y²

# But we need to find the actual pattern
# Let's look for Piecewise that generates 1/Y terms
piecewise_patterns = [
    r'\(\(1e-15 < BCC_A20TI\) \? \([^)]*\) : 0\)',
    r'Piecewise\([^)]*BCC_A20TI[^)]*\)',
    r'log\(BCC_A20TI\)',
]

for pattern in piecewise_patterns:
    matches = re.findall(pattern, s)
    if matches:
        print(f"\nFound {len(matches)} matches for pattern: {pattern}")
        for i, match in enumerate(matches[:3]):
            print(f"  Match {i}: {match[:100]}...")

# The issue might be that the terms are generated during variable conversion
# Let's see what happens after conversion
from pycalphad.gpu.gpu_codegen import notebook_convert_var_names

wks = Workspace(database=db, components=comps, phases=['BCC_A2'], conditions={v.T: 1800, v.P: 101325, v.X('TI'): 0.3})
s_converted = notebook_convert_var_names(s, model, wks)

print("\n\nAfter variable conversion:")
# Look for the spurious entropy terms
entropy_pattern = r'8\.3145\*x\[2\]\*\(1\.0\*\(\(1e-15 < x\[4\]\) \? \(pow\(x\[4\], \(-1\)\)\) : 0\)'
matches = re.findall(entropy_pattern, s_converted)
print(f"\nFound {len(matches)} entropy terms with 1/x[4]")

# Let's see where these come from
print("\n\nTracing the source of 1/Y terms...")
# In SymEngine/SymPy, log(Y) might be represented as Piecewise to handle Y=0
# When differentiated, it becomes Piecewise(1/Y, Y>0, 0)

# Check if model.G has log terms
G_str = str(model.G)
if 'log(' in G_str:
    print("Model.G contains log terms")
    # Extract log terms with BCC_A2 variables
    log_terms = re.findall(r'log\([^)]*BCC_A2[^)]*\)', G_str)
    print(f"Found {len(log_terms)} log terms with BCC_A2 variables")
    for term in log_terms[:5]:
        print(f"  {term}")

# The entropy contribution is R*T*sum(Y*log(Y))
# Its second derivative is R*T/Y for diagonal elements
print("\n\nThe spurious 1/Y terms come from entropy contribution differentiation")
print("For Y_NB, the diagonal Hessian should NOT contain 1/Y_TI terms")