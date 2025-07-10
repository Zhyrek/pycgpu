#!/usr/bin/env python3
"""Analyze e_matrix difference between CPU and GPU"""

print("=== Analyzing e_matrix Calculation ===")

print("\nFrom the debug outputs, I notice that CPU shows:")
print("  c_G values: [ 0.48612027 -0.48612027]")

print("\nBut GPU shows:")
print("  full_e_matrix[0,0]=-3.832798e-05")
print("  c_G[0] = -1.664287e-01")

print("\nThe signs and magnitudes are very different!")

print("\nLet me check if the e_matrix calculation itself is different...")
print("\nThe e_matrix is the inverse of the phase_matrix.")
print("In the CPU code, it's calculated in compute_phase_matrix().")

print("\nKey insight: The CPU c_G values (0.486...) are about 3x larger than GPU (-0.166...)")
print("And they have opposite signs!")

print("\nThis suggests either:")
print("1. The e_matrix calculation is different")
print("2. The gradient values are different")
print("3. There's a sign convention difference")

print("\nFrom the GPU debug output, I can see:")
print("  Row 0: -3.832798e-05 3.832798e-05 5.000000e-01")
print("  Row 1: 3.832798e-05 -3.832798e-05 5.000000e-01")
print("  Row 2: 5.000000e-01 5.000000e-01 -4.154566e+04")

print("\nThis is the INVERTED matrix (full_e_matrix), not the original phase_matrix!")