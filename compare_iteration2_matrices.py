#!/usr/bin/env python
"""Compare CPU and GPU matrices at iteration 2 (post-consolidation)."""

print("COMPARING ITERATION 2 MATRICES (POST-CONSOLIDATION)")
print("=" * 60)

print("\nGPU Matrix at iteration 2 (from debug output):")
print("Row 0: +9.703930e-02 +9.029607e-01 +0.000000e+00 | RHS: -1.991128e+04")
print("Row 1: -3.242196e-05 +3.242196e-05 -1.541164e-16 | RHS: +2.063107e-01")
print("Row 2: +6.691692e-20 -2.159947e-19 +1.000000e+00 | RHS: +1.064380e-15")

print("\nCPU Matrix at iteration 2:")
print("Need to find from debug output...")

print("\n" + "="*60)
print("ANALYSIS OF GPU MATRIX:")

# Row 0: Phase gradient row
print("\nRow 0 (phase gradient):")
print("- Coefficients: [0.0970393, 0.9029607] (site fractions Y_NB, Y_TI)")
print("- RHS: -19911.28 (negative energy gradient)")

# Row 1: Mass balance constraint
print("\nRow 1 (mass balance):")
print("- Coefficients: [-3.242196e-05, +3.242196e-05] (c_G values)")
print("- RHS: 0.2063107")
print("- This is trying to enforce X(TI) = 0.9")

# Row 2: Site fraction sum constraint
print("\nRow 2 (site fraction sum):")
print("- Coefficients: [~0, ~0, 1.0] (for phase amount)")
print("- RHS: ~0")

print("\n" + "="*60)
print("KEY OBSERVATIONS:")

print("\n1. The site fractions in Row 0 are [0.0970393, 0.9029607]")
print("   - This gives X(TI) = 0.9029607 (not 0.903147!)")
print("   - The GPU has ALREADY updated from 0.903147 to 0.902960")

print("\n2. The mass balance RHS is 0.2063107")
print("   - This is the residual: 0.9 - 0.9029607 * 1.0 ≈ -0.0029607")
print("   - But the RHS should be: target - current = 0.9 - 0.9029607 = -0.0029607")
print("   - The actual RHS 0.2063107 seems way too large!")

print("\n3. The c_G values [-3.242196e-05, +3.242196e-05] are different from before")

print("\n" + "="*60)
print("HYPOTHESIS:")
print("The GPU is constructing the wrong RHS for the mass balance constraint")
print("It should be: target - sum(phase_amt * X_phase) = 0.9 - 0.902960 = -0.002960")
print("But it's showing: 0.2063107 (way too large)")

print("\nThis explains why GPU can't converge to 0.9:")
print("- Wrong RHS → wrong solution → wrong site fraction updates")
print("- The system thinks it needs a huge correction when it's already close")