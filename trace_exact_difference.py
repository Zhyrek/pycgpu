#!/usr/bin/env python3
"""Trace the exact difference between CPU and GPU hessian generation"""

import symengine as se

# Create symbols
Y1 = se.Symbol('Y1')
Y2 = se.Symbol('Y2')
T = se.Symbol('T')
R = 8.3145

print("=== The Key Structure ===")
print("Model.G contains: (Y1 + Y2) * [... + RT*(Y1*log(Y1) + Y2*log(Y2))/(Y1 + Y2) + ...]")

# Simplified entropy term as it appears in the model
S_divided = R * T * (Y1 * se.log(Y1) + Y2 * se.log(Y2)) / (Y1 + Y2)
G = (Y1 + Y2) * S_divided

print(f"\nS_divided = {S_divided}")
print(f"G = {G}")
print(f"G expanded = {se.expand(G)}")

# Now differentiate
print("\n=== Differentiation ===")

# First derivative
dG_dY1 = G.diff(Y1)
print(f"\ndG/dY1 = {dG_dY1}")

# Second derivative
d2G_dY1 = dG_dY1.diff(Y1)
print(f"\nd²G/dY1² = {d2G_dY1}")

# Now let's manually evaluate at Y1=0.6, Y2=0.4, T=1000
vals = {Y1: 0.6, Y2: 0.4, T: 1000}
d2G_val = float(d2G_dY1.subs(vals))
print(f"\nAt Y1=0.6, Y2=0.4, T=1000: d²G/dY1² = {d2G_val:.2f}")

# Expected value (just RT/Y1)
expected = R * 1000 / 0.6
print(f"Expected (RT/Y1) = {expected:.2f}")

# The issue: symengine doesn't automatically simplify
# (Y1 + Y2) * expr / (Y1 + Y2) -> expr

print("\n=== The Answer ===")
print("The CPU and GPU both start with the same expression.")
print("The difference must be in:")
print("1. How Piecewise is handled during differentiation")
print("2. Or how the generated C code evaluates the expression")
print("3. Or there's an additional simplification step somewhere")

# Let's check with Piecewise
print("\n=== With Piecewise Protection ===")
S1_pw = se.Piecewise((Y1 * se.log(Y1), Y1 > 1e-15), (0, True))
S2_pw = se.Piecewise((Y2 * se.log(Y2), Y2 > 1e-15), (0, True))
S_divided_pw = R * T * (S1_pw + S2_pw) / (Y1 + Y2)
G_pw = (Y1 + Y2) * S_divided_pw

d2G_pw = G_pw.diff(Y1).diff(Y1)
print(f"\nWith Piecewise: d²G/dY1² has {len(str(d2G_pw))} characters")

# The Piecewise makes the expression much more complex
# This complexity might hide where the difference comes from