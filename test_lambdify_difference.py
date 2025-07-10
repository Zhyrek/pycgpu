#!/usr/bin/env python3
"""Test if lambdify simplifies the expression differently"""

import symengine as se
from symengine import lambdify
import numpy as np

# Create test expression that mirrors the issue
Y1 = se.Symbol('Y1')
Y2 = se.Symbol('Y2')
T = se.Symbol('T')
R = 8.3145

# Expression with entropy divided by sum (like in the model)
S_divided = R * T * se.Piecewise((Y1 * se.log(Y1), Y1 > 1e-15), (0, True)) / (Y1 + Y2)
G = (Y1 + Y2) * S_divided

print("=== Test Expression ===")
print(f"S_divided = {S_divided}")
print(f"G = {G}")
print(f"G expanded = {se.expand(G)}")

# Compute hessian symbolically
d2G_dY1 = G.diff(Y1).diff(Y1)
print(f"\nSymbolic d²G/dY1²: {d2G_dY1}")

# Now lambdify it (like CPU does)
print("\n=== Testing Lambdify ===")

# Simple case first
vars = [Y1, Y2, T]
hess_func = lambdify(vars, [d2G_dY1])

# Test at Y1=0.6, Y2=0.4, T=1000
result = hess_func([0.6, 0.4, 1000.0])
print(f"\nLambdified result at Y1=0.6, Y2=0.4, T=1000: {result:.2f}")
print(f"Expected (RT/Y1): {R * 1000 / 0.6:.2f}")

# Test with sum != 1
result2 = hess_func([0.6, 0.3, 1000.0])
print(f"\nLambdified result at Y1=0.6, Y2=0.3, T=1000: {result2:.2f}")
print(f"Expected (RT/Y1): {R * 1000 / 0.6:.2f}")

# The key test: Does lambdify preserve the /(Y1+Y2) factor?
print("\n=== Testing Specific Terms ===")

# Create a simple test with the problematic structure
test_expr = R * T / Y1 / (Y1 + Y2)
test_func = lambdify([Y1, Y2, T], [test_expr])

val1 = test_func([0.6, 0.4, 1000.0])
val2 = test_func([0.6, 0.3, 1000.0])

print(f"\nRT/Y1/(Y1+Y2) at Y1=0.6, Y2=0.4: {val1:.2f}")
print(f"RT/Y1/(Y1+Y2) at Y1=0.6, Y2=0.3: {val2:.2f}")
print(f"Ratio: {val2/val1:.3f} (should be 1.111 if /(Y1+Y2) is preserved)")

# Check what the actual model's hessian looks like
print("\n=== The Key Insight ===")
print("The CPU lambdify DOES preserve the /(Y1+Y2) term.")
print("But the CPU still gives RT/Y1, not RT/Y1/(Y1+Y2).")
print("This means something else is happening...")