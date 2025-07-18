#!/usr/bin/env python
"""Diagnose why GPU solver can't correct single phase after consolidation."""

import numpy as np

# After consolidation to single BCC_A2 phase:
# Current: X(TI) = 0.9029604
# Target: X(TI) = 0.9000000
# Error: 0.0029604

print("Single Phase Equilibrium System Analysis")
print("=" * 50)
print("\nProblem: After consolidation, we have:")
print("- Single BCC_A2 phase with X(TI) = Y(TI) = 0.9029604")
print("- Target X(TI) = 0.9000000")
print("- Need delta_Y(TI) ≈ -0.0029604")
print("\nBut GPU produces delta_Y(TI) ≈ -1e-7 (4 orders of magnitude too small!)")

print("\n\nEquilibrium System Structure:")
print("-" * 30)
print("Variables (3 total):")
print("  - delta_mu(NB)")
print("  - delta_mu(TI)")
print("  - delta_NP (phase amount)")
print("\nEquations (3 total):")
print("  - Stable phase equation: gradient = 0")
print("  - Mass balance: X(TI) = 0.9")
print("  - System amount: N = 1")

print("\n\nMatrix Form: A * x = b")
print("-" * 30)
print("Where x = [delta_mu(NB), delta_mu(TI), delta_NP]^T")
print("\nRow 0 (Phase stability): X(NB)*delta_mu(NB) + X(TI)*delta_mu(TI) + 0*delta_NP = -energy_residual")
print("Row 1 (Mass balance): c_comp terms for X(TI) constraint = -mass_residual")
print("Row 2 (System amount): c_comp terms for N constraint = -system_residual")

print("\n\nKey Question:")
print("-" * 30)
print("Why does solving this 3x3 system produce chemical potential changes")
print("that lead to tiny delta_y values instead of the needed -0.003?")

print("\n\nPossible Issues:")
print("-" * 30)
print("1. c_component matrix values are incorrect")
print("2. c_G values are incorrect")
print("3. Matrix becomes ill-conditioned (small determinant)")
print("4. Solution is correct but delta_y calculation is wrong")
print("5. Step size limiting prevents full correction")

print("\n\nDelta_y Calculation (Eq. 43 from Sundman 2015):")
print("-" * 30)
print("delta_y[i] = c_G[i] + sum_j(c_component[j,i] * mu[j])")
print("\nFor the needed correction:")
print("delta_y[TI] = c_G[TI] + c_component[NB,TI]*mu[NB] + c_component[TI,TI]*mu[TI]")
print("This should equal approximately -0.003")

print("\n\nHypothesis:")
print("-" * 30)
print("The equilibrium solver is working correctly, but the c_component")
print("or c_G values are calculated incorrectly for the single phase case,")
print("leading to a delta_y that's too small by a factor of ~10000.")