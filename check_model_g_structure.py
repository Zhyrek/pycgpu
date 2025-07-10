#!/usr/bin/env python3
"""Check the exact structure of model.G to understand why GPU doesn't simplify"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import symengine as se

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Model.G Structure ===")
print(f"Type: {type(model.G)}")
print(f"Number of args: {len(model.G.args) if hasattr(model.G, 'args') else 'N/A'}")

if hasattr(model.G, 'args'):
    print("\nArguments:")
    for i, arg in enumerate(model.G.args):
        print(f"  Arg {i}: {type(arg).__name__}")
        if hasattr(arg, 'args') and len(arg.args) < 5:
            print(f"    Sub-args: {[type(a).__name__ for a in arg.args]}")

# Check if it's a Mul with (Y1+Y2) as first factor
G_str = str(model.G)
if G_str.startswith("1.0*(BCC_A20NB + BCC_A20TI)*"):
    print("\nFOUND: G starts with (Y_NB + Y_TI) factor")
    
    # Check if there's a division by (Y_NB + Y_TI) somewhere
    if "/(BCC_A20NB + BCC_A20TI)" in G_str:
        print("FOUND: G contains /(Y_NB + Y_TI) division")
        
        # Count occurrences
        count = G_str.count("/(BCC_A20NB + BCC_A20TI)")
        print(f"Number of /(Y_NB + Y_TI) occurrences: {count}")

print("\n=== Understanding the Issue ===")

# The model.G might have a structure like:
# (Y1 + Y2) * (mechanical_mixture + entropy_term + ...)
# where entropy_term = RT*(Y1*log(Y1) + Y2*log(Y2))/(Y1+Y2)

# But if the expression is built in pieces, symengine might not see the
# opportunity to cancel (Y1+Y2) with 1/(Y1+Y2)

print("\n=== Testing Manual Simplification ===")

# Try to extract and simplify parts
Y_NB = se.Symbol('BCC_A20NB')
Y_TI = se.Symbol('BCC_A20TI')

# Check if we can force simplification
try:
    from symengine import simplify
    G_simplified = simplify(model.G)
    print(f"\nSimplified G same as original? {G_simplified == model.G}")
    
    # Check if simplification removes the divisions
    G_simp_str = str(G_simplified)
    simp_count = G_simp_str.count("/(BCC_A20NB + BCC_A20TI)")
    print(f"/(Y_NB + Y_TI) in simplified: {simp_count}")
except:
    print("\nCouldn't simplify G")

print("\n=== The GPU Issue ===")
print("The GPU likely receives model.G with the structure:")
print("(Y1+Y2) * [terms including entropy/(Y1+Y2)]")
print("\nBut when it converts to C code, the simplification doesn't happen")
print("because the expression is processed piece by piece, not as a whole.")

# Let's check the exact entropy term structure
import re
entropy_pattern = r'8\.3145\*T\*\([^)]+\)\/\(BCC_A20NB \+ BCC_A20TI\)'
entropy_matches = re.findall(entropy_pattern, G_str)
if entropy_matches:
    print(f"\nFound {len(entropy_matches)} entropy terms with /(Y_NB + Y_TI)")
    print(f"Example: {entropy_matches[0][:100]}...")