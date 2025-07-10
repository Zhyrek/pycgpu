#!/usr/bin/env python3
"""Find where GPU differentiation goes wrong"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import symengine as se

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== The Key Discovery ===")
print("When symengine computes (Y1+Y2) * expr/(Y1+Y2), it automatically simplifies to expr")
print("This happens BEFORE differentiation in the CPU code.")

# Let's trace what the GPU does
print("\n=== GPU Code Generation Process ===")

# The GPU starts with model.G
G = model.G
print(f"1. GPU starts with model.G")
print(f"   Type: {type(G)}")

# Check if it's already simplified
G_str = str(G)
if G_str.startswith("1.0*(BCC_A20NB + BCC_A20TI)*"):
    print("2. G has the (Y_NB + Y_TI) factor at the front")

# The GPU then differentiates this
print("\n3. GPU differentiates G to get gradient and hessian")

# But here's the issue: The GPU might be preventing the simplification
print("\n=== The Problem ===")
print("The GPU code generation might be:")
print("1. Converting the expression to string form before differentiation")
print("2. Or handling Piecewise in a way that prevents simplification")
print("3. Or the symbolic expression isn't getting simplified before differentiation")

# Let's check the actual model.G more carefully
print("\n=== Checking Model.G Structure ===")

# If model.G is a Mul (multiplication), check its args
if hasattr(G, 'args'):
    print(f"G has {len(G.args)} arguments:")
    for i, arg in enumerate(G.args):
        print(f"  Arg {i}: {type(arg).__name__} - {str(arg)[:100]}...")

# The issue might be that the (Y_NB + Y_TI) factor and the 1/(Y_NB + Y_TI) 
# aren't adjacent in the expression tree, preventing automatic cancellation

print("\n=== Hypothesis ===")
print("The CPU's symengine automatically simplifies (Y1+Y2)*f/(Y1+Y2) -> f")
print("But the GPU's expression might have these terms separated by other factors,")
print("preventing the automatic simplification.")

# Let's manually check if simplification would help
Y_NB = se.Symbol('BCC_A20NB')
Y_TI = se.Symbol('BCC_A20TI')

# Create a test expression that doesn't simplify automatically
# This happens when terms are nested in Piecewise
entropy = se.Piecewise((Y_NB * se.log(Y_NB), Y_NB > 1e-15), (0, True))
G_test = (Y_NB + Y_TI) * (8.3145 * 1000 * entropy / (Y_NB + Y_TI))

print(f"\n=== Test Case ===")
print(f"G_test = {G_test}")
print(f"Does it simplify? {G_test == 8.3145 * 1000 * entropy}")

# The Piecewise might be preventing simplification!