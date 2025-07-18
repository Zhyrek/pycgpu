#!/usr/bin/env python
"""Trace the equilibrium matrix solution to understand small delta_y."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

print("TRACING EQUILIBRIUM MATRIX SOLUTION")
print("=" * 60)

# From the verbose output, we know:
# - Single phase after consolidation
# - Y(NB) ≈ 0.097, Y(TI) ≈ 0.903
# - Target X(TI) = 0.9
# - Mass residual ≈ 0.003

print("\nExpected Newton update:")
print("- Current X(TI) = 0.903")
print("- Target X(TI) = 0.900")
print("- Required ΔX(TI) = -0.003")
print("- For single sublattice: ΔY(TI) = ΔX(TI) = -0.003")
print("- Since Y(TI) = 1 - Y(NB), we need ΔY(NB) = +0.003")

print("\nBut GPU gives:")
print("- delta_y ≈ 3e-07 (10,000x too small!)")

print("\nPossible issues in the matrix solution:")
print("1. The equilibrium matrix coefficient for dX/dY might be wrong")
print("2. The RHS residual might be scaled incorrectly")  
print("3. The solution vector might be interpreted wrong")
print("4. c_G might not represent what we think it does")

print("\n" + "=" * 60)
print("ANALYSIS OF c_G CALCULATION")
print("=" * 60)

print("\nIn the GPU code, delta_y is calculated as:")
print("  delta_y[i] = c_G[i] + contributions from chemical potentials")

print("\nThe c_G values come from solving the equilibrium matrix.")
print("In the single-phase case, c_G should contain the Newton")
print("update for the site fractions.")

print("\nFor our case with 1 phase and 1 constraint:")
print("- Equilibrium matrix is 3x3")
print("- Solution vector has [Δμ_NB, Δμ_TI, ΔY_NB]")
print("- c_G[0] should be approximately +0.003")

print("\nBut c_G[0] ≈ 3e-07, suggesting:")
print("1. The matrix is nearly singular")
print("2. The constraint row has wrong coefficients")
print("3. The solution is being damped/scaled")

print("\n" + "=" * 60)
print("HYPOTHESIS: Wrong coefficient in constraint row")
print("=" * 60)

print("\nThe mole fraction constraint row should have:")
print("- Coefficient for Y_NB in column 2: dX(TI)/dY(NB) = -1.0")
print("- RHS after residual subtraction: -0.003")

print("\nIf the coefficient is wrong (e.g., very small), then")
print("the solution will also be very small.")

print("\nNeed to check:")
print("1. How mass_jac is calculated")
print("2. How it's used in write_row_fixed_mole_fraction")
print("3. Whether the prefactor is being applied correctly")