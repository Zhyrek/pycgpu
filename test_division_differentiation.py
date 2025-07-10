#!/usr/bin/env python3
"""Test differentiation of expressions with division by site fraction sum"""

import symengine as se

# Create symbols
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

print("=== Testing Differentiation with Division ===")

# Test 1: Expression similar to what appears in the hessian
# This mimics mechanical mixture terms
expr1 = (Y_NB * 100 + Y_TI * 200) / (Y_NB + Y_TI)
print(f"\nExpression 1: {expr1}")

d1_expr1 = expr1.diff(Y_NB)
d2_expr1 = d1_expr1.diff(Y_NB)

print(f"First derivative d/dY_NB: {d1_expr1}")
print(f"Second derivative d²/dY_NB²: {d2_expr1}")

# Check for denominators
if '(Y_NB + Y_TI)**2' in str(d2_expr1) or '(Y_NB + Y_TI)**3' in str(d2_expr1):
    print("  -> Contains (Y_NB + Y_TI)^2 or ^3 in denominator!")

# Test 2: The actual G expression structure
# G = (Y_NB + Y_TI) * mechanical_mixture
# where mechanical_mixture = (Y_NB * g_NB + Y_TI * g_TI) / (Y_NB + Y_TI)
# So G = Y_NB * g_NB + Y_TI * g_TI

g_NB = 100 + 50*T  # Simple test function
g_TI = 200 + 30*T  # Simple test function

# The mechanical mixture form
mech_mix = (Y_NB * g_NB + Y_TI * g_TI) / (Y_NB + Y_TI)
G_mech = (Y_NB + Y_TI) * mech_mix

print(f"\n\nMechanical mixture: {mech_mix}")
print(f"G (mechanical form): {G_mech}")
print(f"G (expanded): {se.expand(G_mech)}")

# Differentiate the mechanical form
d1_G = G_mech.diff(Y_NB)
d2_G = d1_G.diff(Y_NB)

print(f"\nFirst derivative of G: {d1_G}")
print(f"Second derivative of G: {d2_G}")

# Expand to see the structure
d2_G_expanded = se.expand(d2_G)
print(f"\nExpanded second derivative: {d2_G_expanded}")

# Test 3: What if we have the exact structure from the GPU code?
# The GPU has terms like: x[3]*energy_expr / (x[3] + x[4])
energy_NB = -8519.353 + 142.045475*T
energy_TI = -1272.064 + 134.71418*T

# This is the structure the GPU seems to be differentiating
G_gpu_style = (Y_NB + Y_TI) * ((Y_NB * energy_NB + Y_TI * energy_TI) / (Y_NB + Y_TI))

print(f"\n\nGPU-style G expression: {G_gpu_style}")
print(f"Expanded: {se.expand(G_gpu_style)}")

# Differentiate
d2_gpu = G_gpu_style.diff(Y_NB).diff(Y_NB)
print(f"\nSecond derivative (GPU style): {d2_gpu}")

# Count occurrences of (Y_NB + Y_TI) in denominator
d2_gpu_str = str(d2_gpu)
count = d2_gpu_str.count('(Y_NB + Y_TI)**')
print(f"\nNumber of (Y_NB + Y_TI)** terms: {count}")