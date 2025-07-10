#!/usr/bin/env python3
"""Check the symbolic hessian expression"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import sympify
import re

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

print("=== Checking Symbolic Hessian ===")

# Get variables
variables = prf.state_variables + model.site_fractions
wrt = sympify(tuple(variables))
graph = sympify(model.G)

# Compute hessian element [3,3] (d²G/dY_NB²)
grad_Y_NB = graph.diff(wrt[3])
hess_33 = grad_Y_NB.diff(wrt[3])
hess_str = str(hess_33)

print(f"d²G/dY_NB² length: {len(hess_str)} characters")

# Look for Y**(-1) patterns
inv_patterns = re.findall(r'BCC_A20[A-Z]+\*\*\(-1(?:\.0)?\)', hess_str)
print(f"\nFound {len(inv_patterns)} Y**(-1) terms")

# Look for (Y_NB + Y_TI)**(-1) patterns
sum_inv_patterns = re.findall(r'\(BCC_A20NB \+ BCC_A20TI\)\*\*\(-[0-9.]+\)', hess_str)
print(f"Found {len(sum_inv_patterns)} (Y_NB + Y_TI)**(-n) terms")

# Look for 8.3145*T*Y**(-1) pattern (which would be RT/Y)
rt_over_y_patterns = re.findall(r'8\.3145\*T[^,]*BCC_A20[A-Z]+\*\*\(-1(?:\.0)?\)', hess_str)
print(f"\nFound {len(rt_over_y_patterns)} RT/Y patterns")

# Count specific entropy-related patterns
print("\n--- Entropy-Related Terms ---")

# The correct hessian should have 8.3145*T/Y_NB
# But the spurious term would be 8.3145*T/Y_TI also appearing

# Count how many times each site fraction appears with **(-1)
nb_inv_count = hess_str.count('BCC_A20NB**(-1')
ti_inv_count = hess_str.count('BCC_A20TI**(-1')

print(f"BCC_A20NB**(-1) appears: {nb_inv_count} times")
print(f"BCC_A20TI**(-1) appears: {ti_inv_count} times")

# The key insight: For d²G/dY_NB², we should only see Y_NB**(-1), not Y_TI**(-1)
if ti_inv_count > 0:
    print("\nFOUND THE ISSUE: Y_TI**(-1) appears in d²G/dY_NB²!")
    print("This creates the spurious RT/Y_TI term.")

# Let's also check if there's a sum term
if '(BCC_A20NB**(-1) + BCC_A20TI**(-1))' in hess_str or \
   '(BCC_A20NB**(-1.0) + BCC_A20TI**(-1.0))' in hess_str:
    print("\nFOUND: (1/Y_NB + 1/Y_TI) term in hessian!")

# Extract a sample around 8.3145
if '8.3145' in hess_str:
    idx = hess_str.find('8.3145')
    context = hess_str[max(0, idx-50):idx+150]
    print(f"\nSample context around 8.3145:\n{context}")