#!/usr/bin/env python
"""Calculate the exact RHS values based on initial phase data."""

# Initial phase data from GPU output
phase_amounts = [0.037009, 0.304922, 0.658069]  # Sum = 1.0
cu_fractions = [0.117749, 0.402889, 0.110616]  # X(CU) in each phase
fe_fractions = [0.345898, 0.119558, 0.381028]  # X(FE) in each phase

# Calculate system-wide mole fractions
system_cu = sum(phase_amounts[i] * cu_fractions[i] for i in range(3))
system_fe = sum(phase_amounts[i] * fe_fractions[i] for i in range(3))

print("=" * 80)
print("RHS CALCULATION FOR CONSTRAINT EQUATIONS")
print("=" * 80)

print("\nInitial phase data:")
for i in range(3):
    print(f"  Phase {i}: NP={phase_amounts[i]:.6f}, X(CU)={cu_fractions[i]:.6f}, X(FE)={fe_fractions[i]:.6f}")

print(f"\nTotal phase amounts: {sum(phase_amounts):.6f} (should be 1.0)")

print(f"\nSystem-wide mole fractions:")
print(f"  X(CU) = {system_cu:.6f}")
print(f"  X(FE) = {system_fe:.6f}")
print(f"  X(AL) = {1 - system_cu - system_fe:.6f} (by difference)")

print(f"\nPrescribed values:")
print(f"  X(CU) prescribed = 0.200000")
print(f"  X(FE) prescribed = 0.300000")
print(f"  X(AL) implied = 0.500000")

print(f"\nConstraint residuals (X_current - X_prescribed):")
cu_residual = system_cu - 0.2
fe_residual = system_fe - 0.3
print(f"  CU residual = {system_cu:.6f} - 0.200000 = {cu_residual:.6f}")
print(f"  FE residual = {system_fe:.6f} - 0.300000 = {fe_residual:.6f}")

print(f"\nRHS values for equilibrium matrix (= -residual = X_prescribed - X_current):")
print(f"  CU RHS = -{cu_residual:.6f} = {-cu_residual:.6f}")
print(f"  FE RHS = -{fe_residual:.6f} = {-fe_residual:.6f}")

print("\n" + "=" * 80)
print("COMPARISON WITH ACTUAL VALUES:")
print("=" * 80)

print(f"\nCPU RHS values (from debug output):")
print(f"  X(CU): -0.038185 (sum of phase contributions)")
print(f"  X(FE): +0.024757 (sum of phase contributions)")

print(f"\nGPU RHS values (from matrix output):")
print(f"  Row 3 (X(CU)): -0.136")
print(f"  Row 4 (X(FE)): +0.122")

print(f"\nCalculated RHS values (from initial phases):")
print(f"  X(CU): {-cu_residual:.6f}")
print(f"  X(FE): {-fe_residual:.6f}")

print("\n*** ANALYSIS ***")
if abs(-cu_residual - 0.136) < 0.01:
    print("✓ GPU RHS values match our calculation!")
    print("  This confirms the GPU is computing residuals correctly.")
else:
    print("✗ GPU RHS values DON'T match our calculation.")
    print("  GPU may be using different initial phase amounts or compositions.")
    
# Check if the GPU values are exactly what we'd expect
print("\nPossible explanation:")
print("If the GPU starts with different initial guesses for site fractions,")
print("it would get different phase compositions and thus different RHS values.")
print("The large RHS values suggest the initial guess is far from equilibrium.")