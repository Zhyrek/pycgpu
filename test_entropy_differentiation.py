#!/usr/bin/env python3
"""Test differentiation with entropy terms that appear in the model"""

import symengine as se

# Create symbols
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

print("=== Testing with Entropy Terms ===")

# The model includes ideal entropy terms like:
# 8.3145 * T * (Y_NB*log(Y_NB) + Y_TI*log(Y_TI))

# Test the full G structure with entropy
energy_NB = -8519.353 + 142.045475*T
energy_TI = -1272.064 + 134.71418*T

# Ideal mixing entropy (simplified)
R = 8.3145
entropy_ideal = R * T * (Y_NB * se.log(Y_NB) + Y_TI * se.log(Y_TI))

# Interaction parameter
L_NB_TI = 26090.6

# Full G expression as it might appear
# Method 1: Pre-multiplied form (what symengine.expand would give)
G_expanded = Y_NB * energy_NB + Y_TI * energy_TI + entropy_ideal + L_NB_TI * Y_NB * Y_TI

print(f"G (expanded form): {G_expanded}")

# Method 2: Mechanical mixture form (what GPU might be using)
# This keeps the (Y_NB + Y_TI) factor explicit
mech_energy = (Y_NB * energy_NB + Y_TI * energy_TI) / (Y_NB + Y_TI)
G_mechanical = (Y_NB + Y_TI) * mech_energy + entropy_ideal + L_NB_TI * Y_NB * Y_TI

print(f"\nG (mechanical form): {G_mechanical}")

# Now differentiate both forms
print("\n=== Differentiating Expanded Form ===")
d2_expanded = G_expanded.diff(Y_NB).diff(Y_NB)
print(f"d²G/dY_NB² (expanded): {d2_expanded}")

print("\n=== Differentiating Mechanical Form ===")
d1_mechanical = G_mechanical.diff(Y_NB)
print(f"dG/dY_NB (mechanical): {d1_mechanical}")

d2_mechanical = d1_mechanical.diff(Y_NB)
print(f"\nd²G/dY_NB² (mechanical): {d2_mechanical}")

# Count (Y_NB + Y_TI) terms
d2_mech_str = str(d2_mechanical)
count_pow2 = d2_mech_str.count('(Y_NB + Y_TI)**2')
count_pow3 = d2_mech_str.count('(Y_NB + Y_TI)**3')
print(f"\nContains (Y_NB + Y_TI)**2: {count_pow2} times")
print(f"Contains (Y_NB + Y_TI)**3: {count_pow3} times")

# Now let's evaluate both at Y_NB=0.6, Y_TI=0.4, T=1000
vals = {Y_NB: 0.6, Y_TI: 0.4, T: 1000}
print("\n=== Numerical Evaluation at Y_NB=0.6, Y_TI=0.4, T=1000 ===")

try:
    val_expanded = float(d2_expanded.subs(vals))
    print(f"d²G/dY_NB² (expanded form): {val_expanded:.2f}")
except:
    print("Could not evaluate expanded form")

try:
    val_mechanical = float(d2_mechanical.subs(vals))
    print(f"d²G/dY_NB² (mechanical form): {val_mechanical:.2f}")
except:
    print("Could not evaluate mechanical form")

# Check the ratio
try:
    if val_expanded != 0:
        ratio = val_mechanical / val_expanded
        print(f"\nRatio mechanical/expanded: {ratio:.3f}")
except:
    pass