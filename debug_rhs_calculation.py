#!/usr/bin/env python3
"""Debug the RHS calculation in detail"""
import numpy as np

# From GPU debug output
print("=== GPU RHS Calculation Debug ===")

# Phase data
phase_amt = [0.340986, 0.659014]
system_amount = 1.0
X_TI_system = 0.4  # Current system mole fraction
prefactor = 1.0  # For TI component

# Phase 0 data
print("\n--- Phase 0 ---")
c_G_0 = [-0.1664287, 0.1664287]
mass_jac_TI_0 = [0, 0, 0, 0, 1]  # For TI component, only Y_TI has coefficient 1
moles_norm_grad_0 = [0, 0, 0, 1, 1]  # Assuming this from CPU pattern

# Calculate rhs_term1 for phase 0
rhs_term1_0 = mass_jac_TI_0[3] * c_G_0[0] + mass_jac_TI_0[4] * c_G_0[1]
print(f"rhs_term1 = {mass_jac_TI_0[3]} * {c_G_0[0]} + {mass_jac_TI_0[4]} * {c_G_0[1]} = {rhs_term1_0}")

# Calculate rhs_term2 for phase 0
rhs_term2_0 = (-X_TI_system * moles_norm_grad_0[3]) * c_G_0[0] + (-X_TI_system * moles_norm_grad_0[4]) * c_G_0[1]
print(f"rhs_term2 = (-{X_TI_system} * {moles_norm_grad_0[3]}) * {c_G_0[0]} + (-{X_TI_system} * {moles_norm_grad_0[4]}) * {c_G_0[1]}")
print(f"         = {-X_TI_system * moles_norm_grad_0[3]} * {c_G_0[0]} + {-X_TI_system * moles_norm_grad_0[4]} * {c_G_0[1]} = {rhs_term2_0}")

# Total RHS contribution from phase 0
rhs_contrib_0 = -prefactor * (phase_amt[0]/system_amount) * (rhs_term1_0 + rhs_term2_0)
print(f"RHS contribution = -{prefactor} * ({phase_amt[0]}/{system_amount}) * ({rhs_term1_0} + {rhs_term2_0})")
print(f"                 = -{prefactor} * {phase_amt[0]/system_amount} * {rhs_term1_0 + rhs_term2_0} = {rhs_contrib_0}")

# Phase 1 data
print("\n--- Phase 1 ---")
c_G_1 = [-0.1725852, 0.1725852]
mass_jac_TI_1 = [0, 0, 0, 0, 1]  # Same structure
moles_norm_grad_1 = [0, 0, 0, 1, 1]  # Same structure

# Calculate rhs_term1 for phase 1
rhs_term1_1 = mass_jac_TI_1[3] * c_G_1[0] + mass_jac_TI_1[4] * c_G_1[1]
print(f"rhs_term1 = {mass_jac_TI_1[3]} * {c_G_1[0]} + {mass_jac_TI_1[4]} * {c_G_1[1]} = {rhs_term1_1}")

# Calculate rhs_term2 for phase 1
rhs_term2_1 = (-X_TI_system * moles_norm_grad_1[3]) * c_G_1[0] + (-X_TI_system * moles_norm_grad_1[4]) * c_G_1[1]
print(f"rhs_term2 = (-{X_TI_system} * {moles_norm_grad_1[3]}) * {c_G_1[0]} + (-{X_TI_system} * {moles_norm_grad_1[4]}) * {c_G_1[1]}")
print(f"         = {-X_TI_system * moles_norm_grad_1[3]} * {c_G_1[0]} + {-X_TI_system * moles_norm_grad_1[4]} * {c_G_1[1]} = {rhs_term2_1}")

# Total RHS contribution from phase 1
rhs_contrib_1 = -prefactor * (phase_amt[1]/system_amount) * (rhs_term1_1 + rhs_term2_1)
print(f"RHS contribution = -{prefactor} * ({phase_amt[1]}/{system_amount}) * ({rhs_term1_1} + {rhs_term2_1})")
print(f"                 = -{prefactor} * {phase_amt[1]/system_amount} * {rhs_term1_1 + rhs_term2_1} = {rhs_contrib_1}")

# Total RHS
print("\n--- Total RHS ---")
total_rhs_from_phases = rhs_contrib_0 + rhs_contrib_1
print(f"Total from phases: {rhs_contrib_0} + {rhs_contrib_1} = {total_rhs_from_phases}")

# Add residual (should be 0 in this case)
residual = X_TI_system - 0.4  # Current - target
print(f"Residual: {X_TI_system} - 0.4 = {residual}")
print(f"Total RHS: {total_rhs_from_phases} - {residual} = {total_rhs_from_phases - residual}")

print(f"\nGPU reported RHS: -0.2727775")
print(f"Calculated RHS: {total_rhs_from_phases - residual}")

# Now let's check what happens if moles_norm_grad is actually [0,0,0,0,1]
print("\n=== Alternative: moles_norm_grad = [0,0,0,0,1] ===")
moles_norm_grad_alt = [0, 0, 0, 0, 1]

# Phase 0 with alternative
rhs_term2_0_alt = (-X_TI_system * moles_norm_grad_alt[3]) * c_G_0[0] + (-X_TI_system * moles_norm_grad_alt[4]) * c_G_0[1]
rhs_contrib_0_alt = -prefactor * (phase_amt[0]/system_amount) * (rhs_term1_0 + rhs_term2_0_alt)
print(f"Phase 0 RHS contribution: {rhs_contrib_0_alt}")

# Phase 1 with alternative
rhs_term2_1_alt = (-X_TI_system * moles_norm_grad_alt[3]) * c_G_1[0] + (-X_TI_system * moles_norm_grad_alt[4]) * c_G_1[1]
rhs_contrib_1_alt = -prefactor * (phase_amt[1]/system_amount) * (rhs_term1_1 + rhs_term2_1_alt)
print(f"Phase 1 RHS contribution: {rhs_contrib_1_alt}")

total_rhs_alt = rhs_contrib_0_alt + rhs_contrib_1_alt - residual
print(f"Total RHS with alternative: {total_rhs_alt}")

# From GPU output, we have actual values:
print("\n=== From GPU Debug Output ===")
print("Phase 0: rhs_term1=1.664287e-01, rhs_term2=-6.657148e-02")
print("Phase 1: rhs_term1=1.725852e-01, rhs_term2=-6.903408e-02")
print("Phase 0 contribution: -5.674986e-02")
print("Phase 1 contribution: -1.137361e-01")
print("Total: -1.704859e-01")

# But the actual RHS is -0.2727775, so there's another -0.1022916 somewhere
print("\nMissing contribution: -0.2727775 - (-0.1704859) = -0.1022916")