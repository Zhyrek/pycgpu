#!/usr/bin/env python
"""Analyze why c_G and chemical potential contributions cancel out."""

print("DELTA_Y CANCELLATION ANALYSIS")
print("=" * 60)

print("\nFrom the verbose output at iteration 2:")
print("delta_y[0] = 2.96e-07")
print("  c_G[0] = +0.209")
print("  Chemical potential contribution = -0.209")
print("  Total: 0.209 - 0.209 ≈ 3e-07")

print("\nThis near-perfect cancellation is suspicious!")

print("\nIn the Newton method for constrained optimization:")
print("The update equation has the form:")
print("  Δy = -H⁻¹ * (∇G - Σλᵢ∇gᵢ)")
print("Where:")
print("  H = Hessian of Lagrangian")
print("  ∇G = gradient of Gibbs energy")
print("  λᵢ = Lagrange multipliers (chemical potentials)")
print("  ∇gᵢ = gradient of constraints")

print("\nFor our single-phase system:")
print("  c_G represents the contribution from ∇G")
print("  Chemical potential term represents Σλᵢ∇gᵢ")

print("\nThe fact that they cancel means:")
print("1. The current point nearly satisfies the KKT conditions")
print("2. But the composition constraint is NOT satisfied")

print("\nThis suggests the issue is in the CONSTRAINT ENFORCEMENT")
print("not in the energy minimization.")

print("\n" + "=" * 60)
print("ROOT CAUSE HYPOTHESIS:")
print("=" * 60)

print("\nThe equilibrium matrix might be missing the direct")
print("contribution from the mole fraction constraint!")

print("\nIn the matrix:")
print("  [H    B^T] [Δy]   [-∇G + B^T*λ]")
print("  [B    0  ] [Δλ] = [-g         ]")

print("\nWhere:")
print("  B = constraint Jacobian (dX/dY)")
print("  g = constraint residual (X - X_target)")

print("\nThe constraint row should directly relate Δy to the")
print("residual, but it seems this connection is missing or")
print("incorrectly scaled.")