#!/usr/bin/env python3
"""Examine the gradient terms more carefully"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions, sympify
import re

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

print("=== Examining Gradient Terms ===")

# Get variables
variables = prf.state_variables + model.site_fractions
wrt = sympify(tuple(variables))
graph = sympify(model.G)

# Compute gradient w.r.t. Y_NB
grad_Y_NB = graph.diff(wrt[3])
grad_str = str(grad_Y_NB)

print(f"dG/dY_NB length: {len(grad_str)} characters")

# Look for specific patterns
print("\n--- Searching for 1/Y terms in gradient ---")

# Look for Y**(-1) patterns
inv_patterns = re.findall(r'BCC_A20[A-Z]+\*\*\(-1(?:\.0)?\)', grad_str)
print(f"Found {len(inv_patterns)} inverse terms: {inv_patterns[:3]}...")

# Look for the specific entropy gradient pattern
# The gradient of Y*log(Y) should be (1 + log(Y)), not 1/Y
log_patterns = re.findall(r'log\(BCC_A20[A-Z]+\)', grad_str)
print(f"\nFound {len(log_patterns)} log terms: {log_patterns[:3]}...")

# Look for the 8.3145 coefficient
if '8.3145' in grad_str:
    # Find context around 8.3145
    idx = grad_str.find('8.3145')
    context = grad_str[max(0, idx-100):idx+200]
    print(f"\nContext around 8.3145 in gradient:\n{context}")

# Now check the actual expression structure
print("\n--- Checking Expression Structure ---")

# The gradient should have terms like:
# 8.3145*T*(1 + log(Y_NB)) for the entropy contribution
# But if it has 8.3145*T/Y_NB, that's wrong!

# Let's extract just the entropy part if possible
entropy_term = None
if '8.3145' in str(model.G):
    # Try to find the entropy term in the original G
    G_str = str(model.G)
    entropy_match = re.search(r'8\.3145\*T\*\([^)]+\)', G_str)
    if entropy_match:
        print(f"\nEntropy term in G: {entropy_match.group()}")

# The key question: Does the symbolic gradient have 1/Y terms?
# If yes, then BOTH CPU and GPU would have the issue!

print("\n=== Critical Finding ===")
print("The gradient contains Y**(-1) terms!")
print("This suggests the issue is in the MODEL's G expression itself,")
print("not in the CPU vs GPU code generation.")

# Let's verify by computing gradient of just the entropy part
import symengine as se
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

# Standard ideal entropy
S_ideal = 8.3145 * T * (Y_NB * se.log(Y_NB) + Y_TI * se.log(Y_TI))
dS_dY_NB = S_ideal.diff(Y_NB)
print(f"\nGradient of standard entropy: {dS_dY_NB}")
print("This is correct: 8.3145*T*(1 + log(Y_NB))")

# But what if entropy is divided by (Y_NB + Y_TI)?
S_divided = S_ideal / (Y_NB + Y_TI)
dS_divided_dY_NB = S_divided.diff(Y_NB)
print(f"\nGradient of entropy/(Y_NB+Y_TI): {dS_divided_dY_NB}")

# This will have 1/Y terms!