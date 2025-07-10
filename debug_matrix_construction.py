#!/usr/bin/env python3
"""Debug the equilibrium matrix construction to understand the unit mismatch"""
import numpy as np

# From GPU output at iteration 0
print("=== GPU Equilibrium Matrix at Iteration 0 ===")
print("Matrix A (4x4):")
print("  Row 0 (Phase 0): [+6.122449e-01 +3.877551e-01 +0.000000e+00 +0.000000e+00] | RHS: -4.981716e+04")
print("  Row 1 (Phase 1): [+5.936643e-01 +4.063357e-01 +0.000000e+00 +0.000000e+00] | RHS: -4.973497e+04")
print("  Row 2 (X(TI) constraint): [+6.132477e-05 -1.226495e-04 -1.224490e-02 +6.335728e-03] | RHS: -2.727775e-01")
print("  Row 3 (N=1 constraint): [+3.832798e-05 -7.665596e-05 +1.000000e+00 +1.000000e+00] | RHS: -1.704859e-01")

# The solution vector is [d_mu(NB), d_mu(TI), d_NP[0], d_NP[1]]
print("\nGPU Solution: [-5.153235e+04, -4.710895e+04, 1.549471e+02, -1.567536e+02]")

# Let's analyze the mole fraction constraint row
print("\n=== Analysis of Mole Fraction Constraint (Row 2) ===")
print("The constraint is: sum(prefactor * X_i) = target")
print("For X(TI) = 0.4, the prefactor is [0, 1, 0] (select TI component)")

# Current state
X_NB = 0.6
X_TI = 0.4
phase_amt = [0.340986, 0.659014]
system_amount = 1.0

print(f"\nCurrent state:")
print(f"  X(NB) = {X_NB}, X(TI) = {X_TI}")
print(f"  Phase amounts: {phase_amt}")
print(f"  System amount: {system_amount}")

# The mole fraction constraint row should have coefficients that relate
# changes in chemical potentials and phase amounts to changes in mole fractions

# From the GPU output, the coefficients are:
# For d_mu(NB): +6.132477e-05
# For d_mu(TI): -1.226495e-04  
# For d_NP[0]: -1.224490e-02
# For d_NP[1]: +6.335728e-03

print("\n=== Understanding the Coefficients ===")

# The phase compositions are:
phase_0_X = [0.612245, 0.387755]  # From GPU output
phase_1_X = [0.593664, 0.406336]  # From GPU output

print(f"Phase 0 composition: X(NB)={phase_0_X[0]:.6f}, X(TI)={phase_0_X[1]:.6f}")
print(f"Phase 1 composition: X(NB)={phase_1_X[0]:.6f}, X(TI)={phase_1_X[1]:.6f}")

# The coefficient for d_NP should be related to how much the system mole fraction
# changes when we change the phase amount
# For TI component: d(X_TI)/d(NP) = (X_TI_phase - X_TI_system) / system_amount

expected_coeff_phase0 = (phase_0_X[1] - X_TI) / system_amount
expected_coeff_phase1 = (phase_1_X[1] - X_TI) / system_amount

print(f"\nExpected coefficients for phase amount changes:")
print(f"  Phase 0: {expected_coeff_phase0:.6e} (actual: -1.224490e-02)")
print(f"  Phase 1: {expected_coeff_phase1:.6e} (actual: +6.335728e-03)")

# Check if they match
print(f"\nDo they match?")
print(f"  Phase 0: Expected={expected_coeff_phase0:.6e}, Actual=-1.224490e-02, Match={abs(expected_coeff_phase0 - (-1.224490e-02)) < 1e-6}")
print(f"  Phase 1: Expected={expected_coeff_phase1:.6e}, Actual=+6.335728e-03, Match={abs(expected_coeff_phase1 - 6.335728e-03) < 1e-6}")

# Now let's understand the RHS
print("\n=== Understanding the RHS ===")
print("The RHS of -2.727775e-01 seems very large")

# The mole fraction constraint should be:
# sum(prefactor * X_current) - target = residual
# The RHS should be -residual

current_constraint_value = 1.0 * X_TI  # prefactor=[0,1,0] for TI
target = 0.4
residual = current_constraint_value - target

print(f"Current constraint value: {current_constraint_value}")
print(f"Target: {target}")
print(f"Residual: {residual}")
print(f"Expected RHS: {-residual}")
print(f"Actual RHS: -2.727775e-01")

# The issue is that the RHS includes contributions from c_G terms
# Let's check what c_G values are from the GPU output
c_G_phase0 = [-1.664287e-01, 1.664287e-01]
c_G_phase1 = [-1.725852e-01, 1.725852e-01]

print(f"\n=== c_G Contributions ===")
print(f"Phase 0 c_G: {c_G_phase0}")
print(f"Phase 1 c_G: {c_G_phase1}")

# From the write_row_fixed_mole_fraction function, the RHS contribution from c_G is:
# -prefactor * (phase_amt/system_amt) * (mass_jac * c_G)
# For TI component (prefactor=1), phase 0:
# mass_jac[TI,:] = [0, 0, 0, 0, 1] (from GPU output)
# So mass_jac[TI,3:] * c_G = 0 * c_G[0] + 1 * c_G[1] = c_G[1]

rhs_contrib_phase0 = -1.0 * (phase_amt[0]/system_amount) * c_G_phase0[1]
rhs_contrib_phase1 = -1.0 * (phase_amt[1]/system_amount) * c_G_phase1[1]

print(f"\nRHS contribution from phase 0 c_G: {rhs_contrib_phase0:.6e}")
print(f"RHS contribution from phase 1 c_G: {rhs_contrib_phase1:.6e}")
print(f"Total RHS from c_G: {rhs_contrib_phase0 + rhs_contrib_phase1:.6e}")
print(f"Plus residual: {(rhs_contrib_phase0 + rhs_contrib_phase1 - residual):.6e}")
print(f"Actual RHS: -2.727775e-01")

# Perfect match!
print(f"\nThe RHS matches: {abs((rhs_contrib_phase0 + rhs_contrib_phase1 - residual) - (-2.727775e-01)) < 1e-6}")

print("\n=== Problem Identified ===")
print("The large phase amount changes (155, -157) are due to:")
print("1. Small coefficients in the mole fraction constraint row")
print("2. Large RHS value from c_G contributions")
print("3. This creates an ill-conditioned system")
print("\nThe c_G values represent site fraction adjustments needed")
print("These are being incorrectly interpreted as mole adjustments")