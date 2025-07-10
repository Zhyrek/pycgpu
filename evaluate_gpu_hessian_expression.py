#!/usr/bin/env python3
"""Evaluate GPU hessian expression to understand the 2.6x factor"""

import numpy as np

# Test conditions from the debug output
T = 1000.0
Y_NB = 0.612245  
Y_TI = 0.387755
sum_Y = Y_NB + Y_TI  # Should be 1.0

print(f"=== Test Conditions ===")
print(f"T = {T}")
print(f"Y_NB = {Y_NB}")
print(f"Y_TI = {Y_TI}")
print(f"Y_NB + Y_TI = {sum_Y}")

# Energy expressions at T=1000K
energy_NB_1000 = -8519.353 + 142.045475*T - 26.4711*T*np.log(T) + 93399.0/T + 0.000203475*T**2 - 3.5012e-07*T**3
energy_TI_1000 = -1272.064 + 134.71418*T - 25.5768*T*np.log(T) + 7208.0/T - 0.000663845*T**2 - 2.78803e-07*T**3

print(f"\n=== Energy Values at T=1000K ===")
print(f"energy_NB = {energy_NB_1000:.2f}")
print(f"energy_TI = {energy_TI_1000:.2f}")

# Ideal entropy contribution to hessian
R = 8.3145
ideal_hess_NB = R * T / Y_NB
ideal_hess_TI = R * T / Y_TI

print(f"\n=== Ideal Entropy Hessian ===")
print(f"d²S_ideal/dY_NB² = {ideal_hess_NB:.2f}")
print(f"d²S_ideal/dY_TI² = {ideal_hess_TI:.2f}")

# Now let's evaluate key terms from the GPU hessian expression
# From line 19 (element [3,3]):

# Term 1: -2.0*(Y_NB*energy_NB + Y_TI*energy_TI)/(Y_NB + Y_TI)^2
weighted_energy = Y_NB * energy_NB_1000 + Y_TI * energy_TI_1000
term1 = -2.0 * weighted_energy / sum_Y**2

print(f"\n=== GPU Hessian Terms (element [3,3]) ===")
print(f"Weighted energy = {weighted_energy:.2f}")
print(f"Term 1: -2.0*weighted_energy/(Y_NB+Y_TI)^2 = {term1:.2f}")

# Term 2: 26090.6*Y_TI/(Y_NB + Y_TI)
L_NB_TI = 26090.6
term2 = L_NB_TI * Y_TI / sum_Y

print(f"Term 2: L*Y_TI/(Y_NB+Y_TI) = {term2:.2f}")

# Term 3: 2.0*energy_NB/(Y_NB + Y_TI)
term3 = 2.0 * energy_NB_1000 / sum_Y

print(f"Term 3: 2.0*energy_NB/(Y_NB+Y_TI) = {term3:.2f}")

# There are additional complex terms involving (Y_NB + Y_TI)^3
# Let's check a simplified version

# The main non-entropy contribution seems to be from the interaction parameter
# For a regular solution model, d²G/dY_NB² should include:
# 1. Ideal entropy: R*T/Y_NB ≈ 13857.5
# 2. Interaction: Some function of L

# Let's check if the issue is related to how the mechanical mixture is handled
print(f"\n=== Mechanical Mixture Analysis ===")

# The GPU seems to keep terms like:
# (Y_NB + Y_TI) * (Y_NB*g_NB + Y_TI*g_TI)/(Y_NB + Y_TI)
# When differentiated, this gives extra (Y_NB + Y_TI) terms

# Simple test: what if (Y_NB + Y_TI) = 0.9 instead of 1.0?
sum_Y_wrong = 0.9
term1_wrong = -2.0 * weighted_energy / sum_Y_wrong**2
term2_wrong = L_NB_TI * Y_TI / sum_Y_wrong

print(f"\nIf (Y_NB + Y_TI) = {sum_Y_wrong}:")
print(f"  Term 1 would be: {term1_wrong:.2f} (ratio: {term1_wrong/term1:.3f})")
print(f"  Term 2 would be: {term2_wrong:.2f} (ratio: {term2_wrong/term2:.3f})")

# The key insight: Since sum_Y should be 1.0, these denominators don't matter
# But the GPU code includes many more complex terms with (Y_NB + Y_TI) in various powers

print(f"\n=== Hypothesis ===")
print("The 2.6x factor likely comes from:")
print("1. Extra terms with (Y_NB + Y_TI) in denominators that shouldn't exist")
print("2. These terms evaluate to finite values when Y_NB + Y_TI = 1.0")
print("3. The symbolic differentiation creates spurious derivatives")

# Let's manually compute what the hessian should be
# For G = Y_NB*g_NB + Y_TI*g_TI + RT*(Y_NB*ln(Y_NB) + Y_TI*ln(Y_TI)) + L*Y_NB*Y_TI
# d²G/dY_NB² = RT/Y_NB + 0 (since g_NB and g_TI are T-dependent only)

expected_hess = ideal_hess_NB  # Just the ideal entropy term for this simple case
print(f"\n=== Expected vs GPU ===")
print(f"Expected d²G/dY_NB² ≈ {expected_hess:.2f} (just ideal entropy)")
print(f"GPU gives ≈ {expected_hess * 2.6:.2f} (2.6x larger)")
print(f"Extra contribution ≈ {expected_hess * 1.6:.2f}")

# This extra contribution must come from the spurious (Y_NB + Y_TI) terms