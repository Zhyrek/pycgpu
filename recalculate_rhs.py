#!/usr/bin/env python
"""Recalculate RHS using actual mass values from GPU output."""

# From GPU output "Masses:" lines
phase_amounts = [0.037009, 0.304922, 0.658069]
# Phase 0 masses: AL=0.536, CU=0.118, FE=0.346
# Phase 1 masses: AL=0.478, CU=0.403, FE=0.120
# Phase 2 masses: AL=0.508, CU=0.111, FE=0.381

cu_fractions = [0.118, 0.403, 0.111]  # CU mass fractions
fe_fractions = [0.346, 0.120, 0.381]  # FE mass fractions

# Calculate system mole fractions
system_cu = sum(phase_amounts[i] * cu_fractions[i] for i in range(3))
system_fe = sum(phase_amounts[i] * fe_fractions[i] for i in range(3))

print("=" * 80)
print("RECALCULATED RHS VALUES")
print("=" * 80)

print("\nPhase data (from GPU Masses output):")
for i in range(3):
    al_frac = 1 - cu_fractions[i] - fe_fractions[i]
    print(f"  Phase {i}: NP={phase_amounts[i]:.6f}, X(AL)={al_frac:.3f}, X(CU)={cu_fractions[i]:.3f}, X(FE)={fe_fractions[i]:.3f}")

print(f"\nSystem mole fractions:")
print(f"  X(CU) = {system_cu:.6f}")
print(f"  X(FE) = {system_fe:.6f}")
print(f"  X(AL) = {1 - system_cu - system_fe:.6f}")

print(f"\nPrescribed values:")
print(f"  X(CU) = 0.200000")
print(f"  X(FE) = 0.300000")
print(f"  X(AL) = 0.500000")

cu_residual = system_cu - 0.2
fe_residual = system_fe - 0.3

print(f"\nResiduals (X_current - X_prescribed):")
print(f"  CU: {system_cu:.6f} - 0.200000 = {cu_residual:.6f}")
print(f"  FE: {system_fe:.6f} - 0.300000 = {fe_residual:.6f}")

print(f"\nRHS values (= -residual):")
print(f"  CU: {-cu_residual:.6f}")
print(f"  FE: {-fe_residual:.6f}")

print(f"\nGPU actual RHS values:")
print(f"  Row 3: -0.136789")
print(f"  Row 4: +0.122240")

print(f"\nDifference:")
print(f"  CU: GPU shows -0.136789, calculated {-cu_residual:.6f}, diff = {abs(-0.136789 - (-cu_residual)):.6f}")
print(f"  FE: GPU shows +0.122240, calculated {-fe_residual:.6f}, diff = {abs(0.122240 - (-fe_residual)):.6f}")

if abs(-0.136789 - (-cu_residual)) < 0.001 and abs(0.122240 - (-fe_residual)) < 0.001:
    print("\n✓ GPU RHS values MATCH our calculation!")
else:
    print("\n✗ GPU RHS values don't match - there's another issue.")