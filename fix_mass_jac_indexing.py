#!/usr/bin/env python
"""Analyze the mass jacobian indexing issue."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

print("MASS JACOBIAN INDEXING ANALYSIS")
print("=" * 60)

print("\nGenerated formulamole_grad output (2 components × 5 DOF = 10 values):")
print("Position 0-4: NB gradients [dN, dP, dT, dY(NB), dY(TI)]")
print("  [0, 0, 0, 1.0, 0]")
print("Position 5-9: TI gradients [dN, dP, dT, dY(NB), dY(TI)]") 
print("  [0, 0, 0, -1.0, 0]")

print("\nThe gradient -1.0 is at position 8, which is:")
print("  Row 1 (TI), Column 3 (dY(NB))")
print("  This means: d(moles(TI))/dY(NB) = -1.0 ✓ CORRECT!")

print("\nBUT the issue is Y(TI) is dependent!")
print("Since Y(TI) = 1 - Y(NB), we only have 1 DOF for site fractions.")
print("The workspace DOF should be: [N, P, T, Y(NB)]")
print("NOT: [N, P, T, Y(NB), Y(TI)]")

print("\n" + "=" * 60)
print("ROOT CAUSE IDENTIFIED:")
print("=" * 60)

print("\n1. The formulamole_grad function outputs gradients for ALL site fractions")
print("   including dependent ones (5 columns).")

print("\n2. But the GPU minimizer only tracks INDEPENDENT site fractions")
print("   in the DOF array (4 columns).")

print("\n3. When accessing mass_jac in write_row_fixed_mole_fraction:")
print("   - It looks for gradient at column (num_statevars + j)")
print("   - For j=0 (first site fraction), this is column 3")
print("   - But the gradient array has the value at column 3 for Y(NB)")
print("   - And at column 3 for TI row, we get d(moles(TI))/dY(NB) = -1.0")

print("\n4. The issue is likely in how the gradient is being copied")
print("   from the formulamole_grad output to mass_jac.")

print("\nSOLUTION:")
print("Either:")
print("1. Fix formulamole_grad to only output gradients for independent DOF")
print("2. Fix the copying logic to skip dependent site fractions")
print("3. Fix the indexing when accessing mass_jac")