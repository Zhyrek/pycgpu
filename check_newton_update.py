#!/usr/bin/env python
"""Check why Newton solver isn't reducing mass residual."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

# Let's understand the problem analytically
print("NEWTON SOLVER ANALYSIS")
print("=" * 60)

# The GPU is stuck with X(TI) ≈ 0.903 instead of 0.900
# Mass residual ≈ 0.00296 (should be < 1e-12)

print("\nProblem:")
print("- Target: X(TI) = 0.900")
print("- Actual: X(TI) ≈ 0.903")
print("- Error: ≈ 0.003")
print("- This error persists across 200 iterations!")

print("\nFor single-phase equilibrium:")
print("- Only 1 free phase with Y(TI) as the site fraction")
print("- Constraint: X(TI) = Y(TI) = 0.9 (for single sublattice)")
print("- Current: Y(TI) ≈ 0.903")

print("\nNewton update equation:")
print("  Δμ = chemical potential change")
print("  ΔNP = phase amount change") 
print("  ΔY = site fraction change")
print("")
print("The equilibrium matrix has form:")
print("  [dG/dμ   dG/dNP   dG/dY  ] [Δμ ]   [-G  ]")
print("  [dNP/dμ  dNP/dNP  dNP/dY ] [ΔNP] = [-0  ]")
print("  [dX/dμ   dX/dNP   dX/dY  ] [ΔY ]   [-res]")
print("")
print("Where res = X(TI) - 0.9 ≈ 0.003")

print("\nFor single phase with NP=1:")
print("- Phase amount is fixed at 1.0")
print("- Only Y(TI) can change")
print("- Constraint row should directly relate ΔY to residual")

print("\nExpected behavior:")
print("- ΔY should be approximately -0.003")
print("- This would correct Y(TI) from 0.903 to 0.900")

print("\nPossible issues:")
print("1. Matrix conditioning - is the system nearly singular?")
print("2. Step size limiting - is ΔY being artificially reduced?")
print("3. Constraint row - is dX/dY correctly set to 1.0?")
print("4. RHS calculation - is the residual sign correct?")

# Look at the verbose output patterns
print("\n" + "=" * 60)
print("KEY OBSERVATIONS FROM VERBOSE OUTPUT:")
print("=" * 60)

print("\n1. Site fraction changes (delta_y):")
print("   - Very small: ~3e-07 instead of ~3e-03")
print("   - This is 10,000x too small!")

print("\n2. Convergence criteria:")
print("   - mass_residual = 2.96e-03 > 1e-12 (FAIL)")
print("   - largest_y_change = 2.96e-07 < 5e-09 (PASS)")
print("   - The y_change tolerance is preventing correction!")

print("\n3. Matrix structure for single phase:")
print("   - Only 3x3 system (μ_NB, μ_TI, Y_NB)")
print("   - Should be well-conditioned")

print("\nROOT CAUSE HYPOTHESIS:")
print("The convergence check is allowing 'convergence' when Y changes")
print("are small, even if the mass balance constraint is not satisfied.")
print("The solver thinks it has converged because Y isn't changing much,")
print("but the constraint residual remains large.")

print("\nSOLUTION:")
print("The mass_residual check should prevent convergence, but something")
print("is preventing the Newton solver from computing the correct ΔY.")
print("This could be in the equilibrium matrix construction or the")
print("step size control logic.")