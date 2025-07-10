#!/usr/bin/env python3
"""Test what happens when differentiating without expanding"""

import symengine as se

# Create symbols
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

print("=== Understanding GPU Differentiation Issue ===")

# The GPU code seems to have this structure based on the generated code:
# It has terms like: -2.0*(x[3]*energy_NB + x[4]*energy_TI)/pow((x[3] + x[4]), 2)

# Let's create a test that mimics this structure
energy_NB = -8519.353 + 142.045475*T
energy_TI = -1272.064 + 134.71418*T

# This is what the GPU might be differentiating
# Notice the explicit division that's not simplified
G_gpu_form = (Y_NB + Y_TI) * (Y_NB * energy_NB + Y_TI * energy_TI) / (Y_NB + Y_TI)

print(f"GPU form (with redundant division): {G_gpu_form}")

# Let's trace through the differentiation step by step
print("\n=== Step-by-step differentiation ===")

# First derivative with respect to Y_NB
d1 = G_gpu_form.diff(Y_NB)
print(f"\ndG/dY_NB: {d1}")

# Check if it has (Y_NB + Y_TI) in denominator
d1_str = str(d1)
if '(Y_NB + Y_TI)**' in d1_str or '/(Y_NB + Y_TI)' in d1_str:
    print("  -> First derivative has (Y_NB + Y_TI) in denominator!")

# Second derivative
d2 = d1.diff(Y_NB)
print(f"\nd²G/dY_NB²: {d2}")

# Count denominators
d2_str = str(d2)
print(f"\nLength of expression: {len(d2_str)} characters")
count2 = d2_str.count('(Y_NB + Y_TI)**2')
count3 = d2_str.count('(Y_NB + Y_TI)**3')
print(f"Contains (Y_NB + Y_TI)**2: {count2} times")
print(f"Contains (Y_NB + Y_TI)**3: {count3} times")

# Now let's see what happens with the actual BCC model structure
# The model has additional complexity with Piecewise functions
print("\n=== Testing with Piecewise (temperature-dependent) ===")

# Simplified piecewise for Nb
energy_NB_piecewise = se.Piecewise(
    (-8519.353 + 142.045475*T, T < 2750),
    (-37669.3 + 271.720843*T, True)
)

# Create the G expression with piecewise
G_piecewise = (Y_NB + Y_TI) * (Y_NB * energy_NB_piecewise + Y_TI * energy_TI) / (Y_NB + Y_TI)

print(f"\nG with Piecewise: {G_piecewise}")

# Differentiate
d2_piecewise = G_piecewise.diff(Y_NB).diff(Y_NB)
print(f"\nd²G/dY_NB² with Piecewise (length: {len(str(d2_piecewise))} chars)")

# The key insight: if the GPU code generator doesn't simplify
# (Y_NB + Y_TI) * expr / (Y_NB + Y_TI) -> expr
# before differentiation, it will get these extra terms

print("\n=== Key Finding ===")
print("The GPU code appears to differentiate the expression")
print("WITHOUT first simplifying (Y_NB + Y_TI) * f / (Y_NB + Y_TI) -> f")
print("This creates spurious (Y_NB + Y_TI) terms in denominators")