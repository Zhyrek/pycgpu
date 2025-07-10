#!/usr/bin/env python3
"""Trace through the mass jacobian indexing"""

print("=== Mass Jacobian Indexing Analysis ===")

print("\nGPU formulamole_grad output array (10 values):")
print("For 2 components × 5 variables = 10 outputs")
print("Layout: [dNB/dN, dNB/dP, dNB/dT, dNB/dY_NB, dNB/dY_TI, dTI/dN, dTI/dP, dTI/dT, dTI/dY_NB, dTI/dY_TI]")
print("Values: [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0]")

print("\nWhen copying to mass_jac:")
print("actual_dof = num_statevars + phase_dof = 3 + 2 = 5")

print("\nFor component 1 (TI), the copy loop does:")
print("for j in range(5):")
print("  mass_jac[1*5 + j] = mass_jac_temp[1*5 + j]")

print("\nSo it copies:")
print("  mass_jac[5] = mass_jac_temp[5] = 0    (d(moles_TI)/dN)")
print("  mass_jac[6] = mass_jac_temp[6] = 0    (d(moles_TI)/dP)")  
print("  mass_jac[7] = mass_jac_temp[7] = 0    (d(moles_TI)/dT)")
print("  mass_jac[8] = mass_jac_temp[8] = 0    (d(moles_TI)/dY_NB)")
print("  mass_jac[9] = mass_jac_temp[9] = 1.0  (d(moles_TI)/dY_TI)")

print("\nThis should give mass_jac[1,:] = [0, 0, 0, 0, 1.0]")
print("But GPU shows mass_jac[1,:] = [0, 0, 0, -1, 1.0]")

print("\n=== The Problem ===")
print("The generated formulamole_grad is correct!")
print("But somehow the GPU is getting -1 instead of 0 for position [1,3]")
print("This suggests:")
print("1. Memory corruption")
print("2. The function is being called differently than expected")
print("3. There's a transformation happening somewhere")

print("\n=== Next Steps ===")
print("Need to add debug output right after formulamole_grad is called")
print("to see the raw values before any processing.")