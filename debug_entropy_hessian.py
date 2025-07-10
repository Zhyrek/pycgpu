#!/usr/bin/env python3
"""Debug how entropy terms are handled in GPU hessian generation"""

import symengine as se

# Create symbols
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')
R = 8.3145

# The ideal entropy expression in the model
S_ideal = R * T * (Y_NB * se.log(Y_NB) + Y_TI * se.log(Y_TI))

print("=== Ideal Entropy Expression ===")
print(f"S_ideal = {S_ideal}")

# Now let's see what happens when this is part of G with the (Y_NB + Y_TI) factor
# The GPU might be treating it as:
# G = (Y_NB + Y_TI) * g_mechanical + S_ideal

# Where g_mechanical includes the entropy divided by (Y_NB + Y_TI)
# This could create the issue

# Test 1: Direct second derivative
d2S_dYNB2 = S_ideal.diff(Y_NB).diff(Y_NB)
print(f"\nd²S/dY_NB² = {d2S_dYNB2}")
# This should give: 8.3145*T/Y_NB

# Test 2: What if entropy is included in mechanical mixture?
# G = (Y_NB + Y_TI) * [(Y_NB*g_NB + Y_TI*g_TI + S_ideal)/(Y_NB + Y_TI)]
# This is wrong but might be what's happening

S_mechanical = S_ideal / (Y_NB + Y_TI)
G_wrong = (Y_NB + Y_TI) * S_mechanical

print(f"\n=== If Entropy is in Mechanical Mixture (WRONG) ===")
print(f"S_mechanical = S_ideal/(Y_NB + Y_TI) = {S_mechanical}")
print(f"G_wrong = (Y_NB + Y_TI) * S_mechanical = {se.expand(G_wrong)}")

# Differentiate this wrong form
d2G_wrong = G_wrong.diff(Y_NB).diff(Y_NB)
print(f"\nd²G_wrong/dY_NB² = {d2G_wrong}")

# Simplify at Y_NB + Y_TI = 1
d2G_wrong_simple = d2G_wrong.subs(Y_TI, 1 - Y_NB)
print(f"\nAt Y_TI = 1 - Y_NB: {se.expand(d2G_wrong_simple)}")

# Test 3: The actual issue might be in how logarithmic terms are handled
# The GPU code has terms like:
# 8.3145*T*(1/Y_NB + 1/Y_TI)/(Y_NB + Y_TI)

print(f"\n=== The Spurious Term ===")
spurious = R * T * (1/Y_NB + 1/Y_TI) / (Y_NB + Y_TI)
print(f"Spurious term: {spurious}")

# When Y_NB + Y_TI = 1:
spurious_at_1 = R * T * (1/Y_NB + 1/Y_TI)
print(f"When Y_NB + Y_TI = 1: {spurious_at_1}")
print(f"This equals: RT/Y_NB + RT/Y_TI")

# This is exactly what we see in the GPU hessian!
print(f"\n=== Origin of the Spurious Term ===")
print("The term 8.3145*x[2]*(1/x[3] + 1/x[4])/(x[3] + x[4])")
print("appears in the GPU code and creates the extra RT/Y_TI contribution.")