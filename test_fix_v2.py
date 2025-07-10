#!/usr/bin/env python3
"""Test the fix_hessian_spurious_terms_v2 function"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import (notebook_replace_piecewise, notebook_replace_exp, 
                                       fix_ternary_operator_precedence, notebook_convert_var_names,
                                       fix_hessian_spurious_terms_v2)
import pycalphad.variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Create model
model = Model(db, comps, 'BCC_A2')

# Get the Hessian expression for Y_NB, Y_NB
Y_NB = model.variables[1]  # Y(BCC_A2,0,NB)
Y_TI = model.variables[2]  # Y(BCC_A2,0,TI)

print("Y_NB:", Y_NB)
print("Y_TI:", Y_TI)
print()

# Calculate second derivative
dG_dYNB = model.G.diff(Y_NB)
d2G_dYNB2 = dG_dYNB.diff(Y_NB)

# Convert to string and apply transformations
s = str(d2G_dYNB2)
print("Original Hessian d²G/dY_NB² (first 200 chars):")
print(s[:200] + "...")
print()

# Apply transformations before variable conversion
s = notebook_replace_piecewise(s)
s = notebook_replace_exp(s)  
s = fix_ternary_operator_precedence(s)

print("After transformations but BEFORE var conversion (first 500 chars):")
print(s[:500] + "...")
print()

# Check for BCC_A20TI terms
import re
bcc_ti_count = s.count('BCC_A20TI')
bcc_nb_count = s.count('BCC_A20NB')
print(f"\nFound {bcc_ti_count} occurrences of BCC_A20TI")
print(f"Found {bcc_nb_count} occurrences of BCC_A20NB")

# Look for the spurious pattern
spurious_pattern = r'1\.0\*\(\(1e-15 < BCC_A20TI\) \? \(pow\(BCC_A20TI, \(-1\)\)\) : 0\)'
matches = re.findall(spurious_pattern, s)
print(f"\nFound {len(matches)} matches of spurious pattern")

# Test the fix function
print("\n=== Testing fix_hessian_spurious_terms_v2 ===")
fixed_s = fix_hessian_spurious_terms_v2(s, 0, 0, Y_NB, Y_NB)

# Check if anything changed
if fixed_s != s:
    print("Fix was applied!")
    bcc_ti_count_after = fixed_s.count('BCC_A20TI')
    print(f"BCC_A20TI occurrences after fix: {bcc_ti_count_after}")
else:
    print("No changes made by fix")

# Now convert variable names and check the result
s_converted = notebook_convert_var_names(fixed_s, model)
print("\n\nAfter variable conversion (first 500 chars):")
print(s_converted[:500] + "...")

# Check for x[4] terms (which would be Y_TI)
x4_pow_count = s_converted.count('pow(x[4], (-1))')
print(f"\nFound {x4_pow_count} occurrences of pow(x[4], (-1)) in final code")