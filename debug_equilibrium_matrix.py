#!/usr/bin/env python
"""Debug the equilibrium matrix construction and solution for single phase."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

print("EQUILIBRIUM MATRIX ANALYSIS FOR SINGLE PHASE")
print("=" * 60)

# For a single BCC_A2 phase with X(TI) constraint
print("\nSystem structure:")
print("- 1 phase (BCC_A2) with NP = 1.0")
print("- 2 components (NB, TI) + 1 vacancy (VA)")
print("- 1 site fraction Y(NB), with Y(TI) = 1 - Y(NB)")
print("- Constraint: X(TI) = 0.9")
print("- Current: X(TI) ≈ 0.903")

print("\nFree variables:")
print("- μ_NB (chemical potential of NB)")
print("- μ_TI (chemical potential of TI)")
print("- Y_NB (site fraction of NB)")

print("\nEquilibrium matrix (3x3):")
print("Row 0: Gibbs energy minimization for the phase")
print("Row 1: Mole fraction constraint for X(TI) = 0.9")
print("Row 2: System amount constraint (N = 1)")

print("\nMatrix structure:")
print("     [μ_NB] [μ_TI] [Y_NB]   RHS")
print("Row0:  ?      ?      ?     = -G")
print("Row1:  ?      ?    dX/dY   = -(X-0.9)")
print("Row2:  ?      ?      ?     = 0")

print("\nFor single sublattice phase:")
print("- X(NB) = Y(NB)")
print("- X(TI) = Y(TI) = 1 - Y(NB)")
print("- dX(TI)/dY(NB) = -1")

print("\nSo the constraint row should have:")
print("- Coefficient for Y_NB: -1.0")
print("- RHS: -(0.903 - 0.9) = -0.003")

print("\nThis means the solution should give:")
print("- ΔY_NB ≈ -0.003")
print("- New Y(NB) = 0.097 - 0.003 = 0.094")
print("- New Y(TI) = 1 - 0.094 = 0.906 (closer to 0.9)")

print("\n" + "=" * 60)
print("HYPOTHESIS: Matrix construction issue")
print("=" * 60)

print("\nThe small delta_y (~3e-07) suggests either:")
print("1. The equilibrium matrix coefficient dX/dY is wrong")
print("2. The RHS residual calculation is wrong")
print("3. The matrix solution is being scaled incorrectly")
print("4. The c_G values used in delta_y calculation are wrong")

print("\nFrom the trace output:")
print("- delta_y[0] = c_G[0] + chemical potential contributions")
print("- c_G comes from the equilibrium matrix solution")
print("- If c_G is ~3e-07, then the matrix solution is giving tiny values")

print("\nThis suggests the equilibrium matrix is not being")
print("constructed correctly for the mole fraction constraint row.")