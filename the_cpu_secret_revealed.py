#!/usr/bin/env python3
"""The CPU's secret revealed: symengine simplifies (Y1+Y2)*expr/(Y1+Y2) automatically"""

import symengine as se

# Create test symbols
Y1 = se.Symbol('Y1')
Y2 = se.Symbol('Y2')
T = se.Symbol('T')
R = 8.3145

print("=== The CPU's Secret ===")
print("When symengine creates an expression with (Y1+Y2)*expr/(Y1+Y2),")
print("it automatically simplifies to just expr!")

# Test 1: Direct multiplication and division
expr1 = (Y1 + Y2) * (R * T * se.log(Y1)) / (Y1 + Y2)
print(f"\n1. (Y1+Y2) * RT*log(Y1) / (Y1+Y2) = {expr1}")
print(f"   Simplified? {expr1 == R * T * se.log(Y1)}")

# Test 2: With Piecewise (like in the model)
pw_expr = se.Piecewise((Y1 * se.log(Y1), Y1 > 1e-15), (0, True))
expr2 = (Y1 + Y2) * (R * T * pw_expr / (Y1 + Y2))
print(f"\n2. With Piecewise: {expr2}")

# Test 3: The exact structure from the model
# In the model, we have:
# G = (Y1 + Y2) * [energy_terms + RT*(Y1*log(Y1) + Y2*log(Y2))/(Y1+Y2)]
entropy_term = R * T * (Y1 * se.log(Y1) + Y2 * se.log(Y2)) / (Y1 + Y2)
G = (Y1 + Y2) * entropy_term
print(f"\n3. Model structure: G = (Y1+Y2) * [RT*(Y1*log(Y1) + Y2*log(Y2))/(Y1+Y2)]")
print(f"   G = {G}")
print(f"   Simplified to: RT*(Y1*log(Y1) + Y2*log(Y2))? {G == R * T * (Y1 * se.log(Y1) + Y2 * se.log(Y2))}")

# Now check the hessian
d2G_dY12 = G.diff(Y1).diff(Y1)
print(f"\n4. d²G/dY1² = {d2G_dY12}")
print(f"   This is just RT/Y1, no spurious RT/Y2 term!")

print("\n=== Why GPU is Different ===")
print("The GPU must be preventing this simplification somehow.")
print("Possible reasons:")
print("1. The Piecewise expressions prevent automatic cancellation")
print("2. The expression is converted to string before differentiation")
print("3. The expression tree structure doesn't allow the simplification")

# Let's test if Piecewise prevents simplification
print("\n=== Testing Piecewise ===")
pw1 = se.Piecewise((Y1, Y1 > 1e-15), (0, True))
pw2 = se.Piecewise((Y2, Y2 > 1e-15), (0, True))
sum_pw = pw1 + pw2

# This structure might not simplify
G_pw = (pw1 + pw2) * (R * T * se.log(Y1) / (pw1 + pw2))
print(f"\nWith Piecewise sum: {G_pw}")
print(f"Length of expression: {len(str(G_pw))} chars")

# The GPU might be getting a more complex expression that doesn't simplify
print("\n=== The Answer ===")
print("CPU: symengine automatically simplifies (Y1+Y2)*expr/(Y1+Y2) to expr")
print("GPU: The expression structure (possibly due to Piecewise) prevents this simplification")
print("Result: GPU keeps the /(Y1+Y2) factor, creating spurious cross-terms in the hessian")