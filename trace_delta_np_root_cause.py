#!/usr/bin/env python3
"""Trace root cause of large delta_NP values"""

print("=== Root Cause Analysis: Large δNP Values ===")
print("\nCPU gets δNP ≈ ±0.17")
print("GPU gets δNP ≈ ±155 (almost 1000x larger!)")
print("\nThe equilibrium system solves: A·x = b")
print("Where x = [δμ₀, δμ₁, δNP₀, δNP₁]")

print("\n=== From Previous Investigation ===")
print("We found that the mass jacobian is wrong:")
print("- CPU: d(moles_TI)/d(Y_NB) = 0")
print("- GPU: d(moles_TI)/d(Y_NB) = -1")
print("\nThis affects the mole fraction constraint row in matrix A")

print("\n=== How Mass Jacobian Affects δNP ===")
print("The mole fraction constraint row has coefficients:")
print("- For phase 0: sum over j of (mass_jac[TI,j] * c_G[j])")
print("- For phase 1: sum over j of (mass_jac[TI,j] * c_G[j])")

print("\nWith CPU's correct mass_jac[TI,:] = [0, 0, 0, 0, 1]:")
print("- Only j=4 (Y_TI) contributes: 1.0 * c_G[4]")

print("\nWith GPU's wrong mass_jac[TI,:] = [0, 0, 0, -1, 1]:")
print("- j=3 (Y_NB) contributes: -1.0 * c_G[3]")
print("- j=4 (Y_TI) contributes: 1.0 * c_G[4]")
print("- Total: -c_G[3] + c_G[4]")

print("\n=== The Key Insight ===")
print("The c_G values are site fraction derivatives of chemical potentials")
print("For a phase with Y_NB ≈ 0.6, Y_TI ≈ 0.4:")
print("- c_G values are typically opposite signs and similar magnitudes")
print("- CPU gets: c_G[4] ≈ -0.5")
print("- GPU gets: -c_G[3] + c_G[4] ≈ -(-0.5) + (-0.5) ≈ 0")

print("\nWhen the matrix coefficient is near zero:")
print("- Small changes in RHS lead to huge changes in solution")
print("- This explains the ±155 values!")

print("\n=== The Fix ===")
print("Fix the mass jacobian calculation so d(moles_TI)/d(Y_NB) = 0")
print("This is NOT a code generation issue - the generated code is correct")
print("The issue is at runtime, suggesting memory corruption or wrong function pointers")