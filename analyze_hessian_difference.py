#!/usr/bin/env python3
"""Analyze the difference between CPU and GPU hessian values"""

import numpy as np

# Values from the test
T = 1000.0
Y_NB = 0.612245
Y_TI = 0.387755
R = 8.3145

# Results
cpu_hess_33 = 13580.347737  # d²G/dY_NB²
cpu_hess_44 = 21442.663538  # d²G/dY_TI²
gpu_hess_33 = 35023.011274  # d²G/dY_NB²

# Calculate expected values
expected_33 = R * T / Y_NB  # Ideal entropy contribution
expected_44 = R * T / Y_TI  # Ideal entropy contribution

print("=== Analysis of Hessian Values ===")
print(f"\nIdeal entropy contributions:")
print(f"RT/Y_NB = {expected_33:.6f}")
print(f"RT/Y_TI = {expected_44:.6f}")
print(f"RT/Y_NB + RT/Y_TI = {expected_33 + expected_44:.6f}")

print(f"\nCPU values:")
print(f"hessian[3,3] = {cpu_hess_33:.6f} (matches RT/Y_NB)")
print(f"hessian[4,4] = {cpu_hess_44:.6f} (matches RT/Y_TI)")

print(f"\nGPU values:")
print(f"hessian[3,3] = {gpu_hess_33:.6f}")
print(f"Difference from CPU = {gpu_hess_33 - cpu_hess_33:.6f}")
print(f"This difference = {gpu_hess_33 - cpu_hess_33:.6f} ≈ RT/Y_TI = {expected_44:.6f}")

print(f"\nObservation:")
print(f"GPU hessian[3,3] = RT/Y_NB + RT/Y_TI")
print(f"                 = {expected_33:.2f} + {expected_44:.2f}")
print(f"                 = {expected_33 + expected_44:.2f}")

# This suggests the GPU is computing:
# d²G/dY_NB² = RT/Y_NB + RT/Y_TI
# instead of just RT/Y_NB

# This could come from terms like:
# RT*(1/Y_NB + 1/Y_TI)/(Y_NB + Y_TI)
# When Y_NB + Y_TI = 1, this becomes RT*(1/Y_NB + 1/Y_TI)

print(f"\n=== Likely Source of Error ===")
print("The GPU expression likely contains a term like:")
print("RT*(1/Y_NB + 1/Y_TI)/(Y_NB + Y_TI)")
print(f"When Y_NB + Y_TI = 1, this evaluates to: {R*T*(1/Y_NB + 1/Y_TI):.2f}")
print(f"Which matches the GPU result: {gpu_hess_33:.2f}")

# Check the mixed entropy term from the earlier analysis
mixed_entropy = R * T * (1/Y_NB + 1/Y_TI) / 1.0
print(f"\nMixed entropy term = {mixed_entropy:.2f}")
print(f"This exactly matches GPU hessian[3,3]!")

print(f"\n=== Conclusion ===")
print("The GPU hessian includes a spurious mixed entropy term")
print("that arises from not simplifying (Y_NB + Y_TI) = 1")
print("before differentiation. This creates cross-terms between")
print("Y_NB and Y_TI that shouldn't exist in the diagonal elements.")