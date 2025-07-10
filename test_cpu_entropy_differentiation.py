#!/usr/bin/env python3
"""Test how CPU handles entropy differentiation"""

import sympy as sp
import symengine as se

# Test with both sympy and symengine to understand the difference

# Symbols
Y_NB_sp = sp.Symbol('Y_NB')
Y_TI_sp = sp.Symbol('Y_TI')
T_sp = sp.Symbol('T')

Y_NB_se = se.Symbol('Y_NB')
Y_TI_se = se.Symbol('Y_TI')
T_se = se.Symbol('T')

R = 8.3145

print("=== Testing Entropy Differentiation ===")

# Test 1: Simple entropy term with Piecewise (sympy)
print("\n--- SymPy (used by CPU) ---")
# CPU uses Piecewise to protect log
entropy_NB_sp = sp.Piecewise(
    (Y_NB_sp * sp.log(Y_NB_sp), Y_NB_sp > 1e-15),
    (0, True)
)
entropy_TI_sp = sp.Piecewise(
    (Y_TI_sp * sp.log(Y_TI_sp), Y_TI_sp > 1e-15),
    (0, True)
)
S_ideal_sp = R * T_sp * (entropy_NB_sp + entropy_TI_sp)

print(f"S_ideal (sympy): {S_ideal_sp}")

# Differentiate
d2S_dYNB2_sp = S_ideal_sp.diff(Y_NB_sp).diff(Y_NB_sp)
d2S_dYNB_dYTI_sp = S_ideal_sp.diff(Y_NB_sp).diff(Y_TI_sp)

print(f"\nd²S/dY_NB² (sympy): {d2S_dYNB2_sp}")
print(f"d²S/dY_NB∂Y_TI (sympy): {d2S_dYNB_dYTI_sp}")

# Test 2: SymEngine (used by GPU)
print("\n--- SymEngine (used by GPU) ---")
# GPU might handle Piecewise differently
entropy_NB_se = se.Piecewise(
    (Y_NB_se * se.log(Y_NB_se), Y_NB_se > 1e-15),
    (0, True)
)
entropy_TI_se = se.Piecewise(
    (Y_TI_se * se.log(Y_TI_se), Y_TI_se > 1e-15),
    (0, True)
)
S_ideal_se = R * T_se * (entropy_NB_se + entropy_TI_se)

print(f"S_ideal (symengine): {S_ideal_se}")

# Differentiate
d2S_dYNB2_se = S_ideal_se.diff(Y_NB_se).diff(Y_NB_se)
d2S_dYNB_dYTI_se = S_ideal_se.diff(Y_NB_se).diff(Y_TI_se)

print(f"\nd²S/dY_NB² (symengine): {d2S_dYNB2_se}")
print(f"d²S/dY_NB∂Y_TI (symengine): {d2S_dYNB_dYTI_se}")

# Test 3: What if we have a mechanical mixture structure?
print("\n--- Mechanical Mixture Structure ---")
# G = (Y_NB + Y_TI) * [(stuff) + S_ideal/(Y_NB + Y_TI)]
# This structure might create cross-terms

# Simple test without Piecewise first
Y1 = se.Symbol('Y1')
Y2 = se.Symbol('Y2')
S_simple = R * T_se * (Y1 * se.log(Y1) + Y2 * se.log(Y2))

# If this is divided by (Y1 + Y2) and then multiplied back
S_mech = S_simple / (Y1 + Y2)
G_mech = (Y1 + Y2) * S_mech

print(f"\nS_simple: {S_simple}")
print(f"S_mechanical: {S_mech}")
print(f"G_mechanical: {se.expand(G_mech)}")

# The key: G_mechanical should simplify back to S_simple
# But if differentiation happens before simplification...
d2G_mech = G_mech.diff(Y1).diff(Y1)
d2S_simple = S_simple.diff(Y1).diff(Y1)

print(f"\nd²G_mech/dY1²: {d2G_mech}")
print(f"d²S_simple/dY1²: {d2S_simple}")
print(f"Are they equal? {d2G_mech == d2S_simple}")

# The issue: If the GPU keeps the mechanical mixture form during differentiation,
# it might get different results!