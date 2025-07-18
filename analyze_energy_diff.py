#!/usr/bin/env python
"""Analyze the energy calculation differences."""

# CPU energies
cpu_phase0 = -19941.00371858601
cpu_phase1 = -19881.67865867988

# GPU energies  
gpu_phase0 = -19941.003719
gpu_phase1 = -19881.678659

# Calculate differences
diff0 = gpu_phase0 - cpu_phase0
diff1 = gpu_phase1 - cpu_phase1

print("ENERGY CALCULATION COMPARISON")
print("=" * 60)
print(f"\nPhase 0:")
print(f"  CPU: {cpu_phase0:.15f} J/mol")
print(f"  GPU: {gpu_phase0:.15f} J/mol")
print(f"  Difference: {diff0:.15e} J/mol")
print(f"  Relative: {abs(diff0/cpu_phase0)*100:.10f}%")

print(f"\nPhase 1:")
print(f"  CPU: {cpu_phase1:.15f} J/mol")
print(f"  GPU: {gpu_phase1:.15f} J/mol")
print(f"  Difference: {diff1:.15e} J/mol")
print(f"  Relative: {abs(diff1/cpu_phase1)*100:.10f}%")

print(f"\nTotal energy difference: {abs(diff0) + abs(diff1):.15e} J/mol")

# Check gradient values too
print("\n" + "=" * 60)
print("GRADIENT COMPARISON")
print("=" * 60)

# From the trace:
# CPU gradient values: [-19434.73793796 -13118.21765729] and [-19787.40464808 -13187.33540866]
# GPU gradient values: [-1.943474e+04, -1.311822e+04] and [-1.978740e+04, -1.318734e+04]

cpu_grad0 = [-19434.73793796, -13118.21765729]
cpu_grad1 = [-19787.40464808, -13187.33540866]

gpu_grad0 = [-1.943474e+04, -1.311822e+04]
gpu_grad1 = [-1.978740e+04, -1.318734e+04]

print("\nPhase 0 gradients:")
print(f"  CPU: {cpu_grad0}")
print(f"  GPU: {gpu_grad0}")
print(f"  Diff[0]: {gpu_grad0[0] - cpu_grad0[0]:.6e}")
print(f"  Diff[1]: {gpu_grad0[1] - cpu_grad0[1]:.6e}")

print("\nPhase 1 gradients:")
print(f"  CPU: {cpu_grad1}")
print(f"  GPU: {gpu_grad1}")
print(f"  Diff[0]: {gpu_grad1[0] - cpu_grad1[0]:.6e}")
print(f"  Diff[1]: {gpu_grad1[1] - cpu_grad1[1]:.6e}")

# Now check which one is earlier - energy or gradient
print("\n" + "=" * 60)
print("ROOT CAUSE ANALYSIS")
print("=" * 60)
print("\nThe FIRST divergence is in the ENERGY CALCULATION (formulaobj)")
print("This happens BEFORE gradient calculation")
print("\nThis suggests the issue is in:")
print("1. The energy model evaluation (formulaobj) itself")
print("2. Different site fraction values being passed to formulaobj")
print("3. Different state variable values (N, P, T)")

# Let's check site fractions from the trace
print("\nNeed to check the DOF values passed to formulaobj...")