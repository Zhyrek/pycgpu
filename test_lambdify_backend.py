#!/usr/bin/env python3
"""Test if LLVM backend simplifies the expression"""

import symengine as se
from symengine import lambdify
import numpy as np

# Create test symbols
Y1 = se.Symbol('Y1')
Y2 = se.Symbol('Y2')
T = se.Symbol('T')
R = 8.3145

# Create the problematic expression structure
# This mimics what happens in the model
G = (Y1 + Y2) * (R * T * (Y1 * se.log(Y1) + Y2 * se.log(Y2)) / (Y1 + Y2))

print("=== Testing Expression Simplification ===")
print(f"G = {G}")

# Differentiate twice w.r.t Y1
dG_dY1 = G.diff(Y1)
d2G_dY12 = dG_dY1.diff(Y1)

print(f"\nd²G/dY1² = {d2G_dY12}")

# Check if it contains spurious terms
d2G_str = str(d2G_dY12)
if 'Y2**(-1)' in d2G_str:
    print("\nFOUND: d²G/dY1² contains Y2**(-1) spurious term!")

# Now test different lambdify backends
vars = [Y1, Y2, T]
test_vals = [0.6, 0.4, 1000.0]

print("\n=== Testing Different Backends ===")

# Test with LLVM backend (CPU default)
try:
    hess_llvm = lambdify(vars, d2G_dY12, backend='llvm')
    result_llvm = hess_llvm(test_vals)
    print(f"\nLLVM backend result: {result_llvm:.2f}")
    print(f"Expected (RT/Y1): {R * 1000 / 0.6:.2f}")
    print(f"Ratio: {result_llvm / (R * 1000 / 0.6):.3f}")
except Exception as e:
    print(f"\nLLVM backend error: {e}")
    result_llvm = None

# Test with lambda backend (pure Python)
hess_lambda = lambdify(vars, d2G_dY12, backend='lambda')
result_lambda = hess_lambda(test_vals)
print(f"\nLambda backend result: {result_lambda:.2f}")
print(f"Ratio: {result_lambda / (R * 1000 / 0.6):.3f}")

# Test with numpy backend
hess_numpy = lambdify(vars, d2G_dY12, backend='numpy')
result_numpy = hess_numpy(test_vals)
print(f"\nNumPy backend result: {result_numpy:.2f}")
print(f"Ratio: {result_numpy / (R * 1000 / 0.6):.3f}")

print("\n=== The Secret ===")
if result_llvm is not None and abs(result_llvm - R * 1000 / 0.6) < 1:
    print("LLVM backend automatically simplifies away the spurious terms!")
else:
    print("LLVM backend does NOT remove spurious terms...")
    
# Let's check if the expression itself simplifies
print("\n=== Checking Symbolic Simplification ===")

# Try to simplify the expression
try:
    from symengine import simplify
    d2G_simplified = simplify(d2G_dY12)
    print(f"Simplified: {d2G_simplified}")
except:
    print("Couldn't simplify expression")

# The real answer might be that when Y1 + Y2 = 1.0 (as it should be),
# the spurious terms cancel out!
print("\n=== Testing with Y1 + Y2 = 1.0 ===")

# Substitute Y2 = 1 - Y1
G_constrained = G.subs(Y2, 1 - Y1)
print(f"G with Y2 = 1 - Y1: {G_constrained}")

# Now differentiate
d2G_constrained = G_constrained.diff(Y1).diff(Y1)
print(f"\nd²G/dY1² with constraint: {d2G_constrained}")

# This should NOT have spurious terms!