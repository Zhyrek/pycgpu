#!/usr/bin/env python
"""Test what solution the CPU matrix gives."""

import numpy as np

# The exact CPU matrix from iteration 2 after consolidation
cpu_matrix = np.array([
    [9.685286e-02, 9.031471e-01, 0.000000e+00],
    [-3.231946e-05, 3.231946e-05, 0.000000e+00],
    [-1.355253e-20, 4.743385e-20, 1.000000e+00]
])

cpu_rhs = np.array([
    -1.991008e+04,
    2.056487e-01,
    -1.204592e-14
])

print("CPU MATRIX ANALYSIS")
print("=" * 60)
print("\nCPU Matrix (3x3):")
for i in range(3):
    print(f"  Row {i}: ", end="")
    for j in range(3):
        print(f"{cpu_matrix[i,j]:+.10e} ", end="")
    print(f"| RHS: {cpu_rhs[i]:+.10e}")

# Analyze the matrix structure
print("\nMatrix structure:")
print("  Row 0: Energy minimization (gradient . delta = -energy)")
print("  Row 1: Mole fraction constraint (X(TI) = 0.9)")
print("  Row 2: System amount constraint (N = 1)")
print("\nUnknowns:")
print("  x[0]: Delta phase amount")
print("  x[1]: Delta chemical potential (NB)")
print("  x[2]: Delta chemical potential (TI)")

# Solve with numpy
x_numpy, residuals, rank, s = np.linalg.lstsq(cpu_matrix, cpu_rhs, rcond=None)
print(f"\nNumPy solution:")
print(f"  x[0] = {x_numpy[0]:+.15e} (delta phase amount)")
print(f"  x[1] = {x_numpy[1]:+.15e} (delta mu_NB)")
print(f"  x[2] = {x_numpy[2]:+.15e} (delta mu_TI)")
print(f"\nSingular values: {s}")
print(f"Condition number: {s[0]/s[-1]:.2e}")

# But wait - the CPU trace showed a tiny update!
print("\nBUT the CPU trace showed:")
print("  [CPU ADVANCE] Phase 0: old=1.000000e+00, delta=-1.147846e-14")
print("  This is x[0] = -1.147846e-14, not -2.5657e+04!")

print("\nThis suggests the CPU is NOT using x[0] directly as delta phase amount.")
print("The large values might be chemical potentials, not phase amounts.")

# Let's check what happens if we interpret the matrix differently
print("\n\nAlternative interpretation:")
print("What if the unknowns are:")
print("  x[0]: Delta chemical potential (NB)")  
print("  x[1]: Delta chemical potential (TI)")
print("  x[2]: Delta phase amount")

# In that case, x[2] = -1.147846e-14 matches the CPU trace!
print(f"\nThis gives delta phase amount = {x_numpy[2]:+.15e}")
print("Which matches the CPU trace exactly!")

# Check the GPU matrix to see if it's structured differently
print("\n\nTo fix the GPU:")
print("1. Check if GPU matrix has same structure as CPU")
print("2. Check if GPU interprets solution vector correctly")
print("3. The issue might be in how the solution is applied, not computed")