#!/usr/bin/env python
"""Extract the EXACT CPU matrix from iteration 2 that achieves X(TI)=0.9."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# From the CPU trace output, I can see the exact matrix at iteration 2:
print("EXACT CPU MATRIX FROM ITERATION 2 (after consolidation)")
print("=" * 60)

# From CPU debug output at iteration 2:
# [CPU MATRIX DEBUG] Equilibrium matrix at iteration 2 (rows=3, cols=3):
#   Row 0: +9.685286e-02 +9.031471e-01 +0.000000e+00 | RHS: -1.991008e+04
#   Row 1: -3.231946e-05 +3.231946e-05 +0.000000e+00 | RHS: +2.056487e-01
#   Row 2: -1.355253e-20 +4.743385e-20 +1.000000e+00 | RHS: -1.204592e-14

cpu_matrix = np.array([
    [9.685286e-02, 9.031471e-01, 0.000000e+00],
    [-3.231946e-05, 3.231946e-05, 0.000000e+00], 
    [-1.355253e-20, 4.743385e-20, 1.000000e+00]
])

cpu_rhs = np.array([
    -1.991008e+04,
    +2.056487e-01,
    -1.204592e-14
])

print("CPU Matrix:")
print(cpu_matrix)
print(f"CPU RHS: {cpu_rhs}")

# Solve this exact system
cpu_solution = np.linalg.solve(cpu_matrix, cpu_rhs)
print(f"CPU solution: {cpu_solution}")

print("\nThis solution represents: [delta_mu_NB, delta_mu_TI, delta_phase_amount]")
print(f"Phase amount change: {cpu_solution[2]:.15e}")

# The current state before this iteration was:
# Phase amount = 1.0, X(TI) = 0.903147
current_phase_amount = 1.0
current_X_TI = 0.903147

new_phase_amount = current_phase_amount + cpu_solution[2]
final_X_TI = new_phase_amount * current_X_TI

print(f"\nAfter applying this update:")
print(f"New phase amount: {new_phase_amount:.15e}")
print(f"Final X(TI): {final_X_TI:.15e}")
print(f"Target X(TI): 0.900000000000000")
print(f"Error: {abs(final_X_TI - 0.9):.2e}")

print(f"\n" + "="*60)
print("NOW TEST IF GPU CONSTRUCTS THE SAME MATRIX")

# Run a simple GPU test to see what matrix it constructs
print("\nWe need to extract the GPU's matrix at the same point.")
print("The GPU should construct the identical 3x3 system after consolidation.")

# Let's check what the key row (mole fraction constraint) should be
print(f"\nAnalyzing the mole fraction constraint row:")
print(f"Row 1: [-3.231946e-05, +3.231946e-05, +0.000000e+00] | RHS: +2.056487e-01")
print(f"This row represents the constraint: delta_contributions = target_change")
print(f"The RHS 0.2056487 comes from c_G contributions minus residual")

# From CPU trace: RHS before residual = 0.208796, residual = 0.003147
# So: 0.208796 - 0.003147 = 0.205649 ≈ 0.2056487
rhs_before_residual = 0.208796
residual = 0.003147
rhs_after_residual = rhs_before_residual - residual
print(f"\nRHS calculation: {rhs_before_residual:.6f} - {residual:.6f} = {rhs_after_residual:.6f}")

print(f"\nThe matrix coefficients [-3.231946e-05, +3.231946e-05] come from c_component matrix")
print(f"These are the derivatives of mole fractions w.r.t. chemical potentials")

print(f"\n" + "="*60)
print("CRITICAL TEST: Run GPU and capture its matrix at iteration 2")
print("If the GPU matrix differs from this exact CPU matrix, that's the bug.")