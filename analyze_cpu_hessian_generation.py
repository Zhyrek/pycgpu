#!/usr/bin/env python3
"""Analyze how CPU generates hessian without spurious terms"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import symengine as se

# Load the model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Analyzing CPU Hessian Generation ===")

# The key is in the model structure
print(f"\nModel.G type: {type(model.G)}")
print(f"Model.G expression (first 500 chars): {str(model.G)[:500]}")

# Check if G has the (Y_NB + Y_TI) factor
G_str = str(model.G)
if '(BCC_A20NB + BCC_A20TI)' in G_str:
    print("\nG has (Y_NB + Y_TI) factor!")

# Let's check how the ideal entropy appears in G
print("\n=== Looking for Entropy Terms ===")

# The ideal entropy should be in the form:
# RT * (Y_NB*log(Y_NB) + Y_TI*log(Y_TI))
# But with Piecewise protection

# Search for log terms
import re
log_matches = re.findall(r'log\([^)]+\)', G_str)
print(f"Found {len(log_matches)} log terms")
for i, match in enumerate(log_matches[:5]):
    print(f"  {i}: {match}")

# Check for Piecewise
piecewise_count = G_str.count('Piecewise')
print(f"\nFound {piecewise_count} Piecewise expressions")

# The key insight: The CPU's build_functions might be doing something special
print("\n=== Hypothesis ===")
print("The CPU might be:")
print("1. Expanding (Y_NB + Y_TI) * expression before differentiation")
print("2. Or using a different code path for ideal entropy")
print("3. Or the C code generation handles Piecewise differently")

# Let's manually check what happens with a simple case
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

# Ideal entropy with Piecewise (simplified)
S_NB = se.Piecewise((Y_NB * se.log(Y_NB), Y_NB > 1e-15), (0, True))
S_TI = se.Piecewise((Y_TI * se.log(Y_TI), Y_TI > 1e-15), (0, True))
S_ideal = 8.3145 * T * (S_NB + S_TI)

print(f"\n=== Simple Test Case ===")
print(f"S_ideal: {S_ideal}")

# Now with mechanical mixture
G_test = (Y_NB + Y_TI) * S_ideal / (Y_NB + Y_TI)
print(f"\nG_test (mechanical): {G_test}")

# The issue is that symengine might not simplify this automatically
# Let's see what the second derivative gives
d2G = G_test.diff(Y_NB).diff(Y_NB)
print(f"\nd²G/dY_NB²: {d2G}")

# The CPU must be avoiding this structure somehow
print("\n=== Conclusion ===")
print("The CPU likely avoids spurious terms by:")
print("1. Not using the mechanical mixture form for entropy")
print("2. Or expanding expressions before differentiation")
print("3. Or having special handling in the C code generation")