#!/usr/bin/env python
"""Analyze the divergence between CPU and GPU at iteration 2."""

print("DIVERGENCE ANALYSIS AT ITERATION 2")
print("=" * 60)

print("\nCPU Matrix at iteration 2:")
print("Row 0: +9.685286e-02 +9.031471e-01 +0.000000e+00 | RHS: -1.991008e+04")
print("Row 1: -3.231946e-05 +3.231946e-05 +0.000000e+00 | RHS: +2.056487e-01")
print("Row 2: +1.355253e-20 -5.421011e-20 +1.000000e+00 | RHS: +4.163336e-15")

print("\nGPU Matrix at iteration 2:")
print("Row 0: +9.703930e-02 +9.029607e-01 +0.000000e+00 | RHS: -1.991128e+04")
print("Row 1: -3.242196e-05 +3.242196e-05 -1.541164e-16 | RHS: +2.063107e-01")
print("Row 2: +6.691692e-20 -2.159947e-19 +1.000000e+00 | RHS: +1.064380e-15")

print("\n" + "="*60)
print("KEY DIFFERENCES:")

print("\n1. Site fractions (Row 0 coefficients):")
print("   CPU: [0.0968529, 0.9031471] → X(TI) = 0.9031471")
print("   GPU: [0.0970393, 0.9029607] → X(TI) = 0.9029607")
print("   Difference: CPU still has 0.903147, GPU already updated to 0.902960!")

print("\n2. c_G values (Row 1 coefficients):")
print("   CPU: [-3.231946e-05, +3.231946e-05]")
print("   GPU: [-3.242196e-05, +3.242196e-05]")
print("   Small difference in c_G values")

print("\n3. RHS values:")
print("   CPU Row 1 RHS: 0.2056487")
print("   GPU Row 1 RHS: 0.2063107")
print("   Similar magnitude, but GPU is slightly larger")

print("\n" + "="*60)
print("ROOT CAUSE IDENTIFIED:")

print("\nThe GPU has ALREADY UPDATED its site fractions before iteration 2!")
print("- CPU at iteration 2: X(TI) = 0.903147 (same as after consolidation)")
print("- GPU at iteration 2: X(TI) = 0.902960 (already changed!)")

print("\nThis means:")
print("1. GPU applied an update between consolidation and iteration 2")
print("2. This update changed X(TI) from 0.903147 to 0.902960")
print("3. CPU did NOT apply this update")

print("\n" + "="*60)
print("HYPOTHESIS:")

print("\nThe divergence happens IMMEDIATELY after consolidation:")
print("- Both consolidate to single phase with X(TI) = 0.903147")
print("- GPU then applies some update that changes it to 0.902960")
print("- CPU keeps it at 0.903147 and solves from there")
print("- This initial divergence causes all subsequent differences")

print("\nNeed to find what happens between:")
print("1. End of consolidation (both have X(TI) = 0.903147)")
print("2. Start of iteration 2 (CPU: 0.903147, GPU: 0.902960)")