#!/usr/bin/env python3
"""Check where entropy contributions come from in the model"""

from pycalphad import Database
from pycalphad.model import Model
from pycalphad.core.workspace import Workspace
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
T = model.variables[0]  # Temperature

print("=== Checking entropy contribution in model.G ===")
G_str = str(model.G)

# Look for entropy terms - they usually appear as T*log(Y) or similar
import re

# Find all log terms
log_terms = re.findall(r'[^(]*log\([^)]+\)[^)]*', G_str)
print(f"\nFound {len(log_terms)} log terms in model.G:")
for i, term in enumerate(log_terms[:5]):
    print(f"  {i}: {term[:100]}...")

# Let's look at the actual model structure
print("\n=== Model energy contributions ===")
if hasattr(model, 'GM'):
    print(f"Mechanical contribution (GM): {model.GM}")
if hasattr(model, 'GE'):
    print(f"Excess contribution (GE): {model.GE}")

# The ideal mixing entropy is usually added separately
# It should be R*T*sum(Y*log(Y))
print("\n=== Ideal mixing entropy ===")
R = 8.3145
ideal_entropy = R * T * (Y_NB * se.log(Y_NB) + Y_TI * se.log(Y_TI))
print(f"Expected ideal entropy: {ideal_entropy}")

# Check if this appears in model.G
if 'log(BCC_A20NB)' in G_str and 'log(BCC_A20TI)' in G_str:
    print("\nIdeal entropy terms found in model.G")
    
    # Check the coefficient
    # Pattern like: 8.3145*T*BCC_A20NB*log(BCC_A20NB)
    entropy_nb_pattern = r'([\d.]+)\*T\*BCC_A20NB\*log\(BCC_A20NB\)'
    match = re.search(entropy_nb_pattern, G_str)
    if match:
        coeff = float(match.group(1))
        print(f"Coefficient for Y_NB entropy: {coeff} (expected: {R})")

# Now let's check what happens when we differentiate
print("\n=== Differentiation of log terms ===")
# d/dY log(Y) = 1/Y
# This is where the 1/Y terms come from

# For the Hessian d²G/dY_NB²:
# The Y_NB*log(Y_NB) term gives: d²/dY_NB²[Y_NB*log(Y_NB)] = 1/Y_NB
# The Y_TI*log(Y_TI) term gives: d²/dY_NB²[Y_TI*log(Y_TI)] = 0 (no Y_NB dependence)

print("\nFor diagonal Hessian d²G/dY_NB²:")
print("- Y_NB*log(Y_NB) term contributes: R*T/Y_NB")
print("- Y_TI*log(Y_TI) term contributes: 0")
print("\nSo there should be NO 1/Y_TI terms in the Y_NB diagonal!")

# The issue might be with the (Y_NB + Y_TI) = 1 constraint
print("\n=== Checking for constraint handling ===")
# If the model uses (Y_NB + Y_TI) in denominators, that could introduce cross-terms
sum_pattern = r'\(BCC_A20NB \+ BCC_A20TI\)'
sum_matches = re.findall(sum_pattern, G_str)
print(f"Found {len(sum_matches)} instances of (Y_NB + Y_TI) in model.G")

if sum_matches:
    print("\nThis could be the source of spurious cross-terms!")
    print("When differentiating expressions with (Y_NB + Y_TI) in denominators,")
    print("we get cross-derivative terms that shouldn't be there.")