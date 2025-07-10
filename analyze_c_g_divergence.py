#!/usr/bin/env python3
"""Analyze c_G divergence between CPU and GPU"""

print("=== Analysis of c_G Values from Debug Output ===")
print("\nFrom the debug output, we can see the c_G values differ:")

print("\nCPU Phase 0 (iteration 0):")
print("  c_G values: [ 0.48612027 -0.48612027]")
print("  This is from the gradient and e_matrix calculation")

print("\nGPU Phase 0 (iteration 0):")
print("  c_G[0] = -1.664286962790051e-01")
print("  c_G[1] = 1.664286962790047e-01")

print("\nDifference:")
print("  CPU c_G[0] = 0.48612027")
print("  GPU c_G[0] = -0.16642870")
print("  Difference = 0.65254897")

print("\nThe GPU c_G values appear to be calculated differently!")
print("Let's check the formulas:")

print("\nFrom GPU debug output:")
print("  c_G[0] calc: full_e_matrix[0,0]=-3.832798e-05 * grad[3]=-4.008941e+04 = 1.536546e+00")
print("  c_G[0] calc: full_e_matrix[0,1]=3.832798e-05 * grad[4]=-3.574718e+04 = -1.370117e+00")
print("  c_G[0] = -1.664287e-01 (sum of above)")

print("\nThe issue: GPU is SUMMING the contributions, not taking individual terms!")
print("CPU uses c_G = e_matrix @ grad[num_statevars:]")
print("GPU incorrectly sums: c_G[0] = e_matrix[0,0]*grad[3] + e_matrix[0,1]*grad[4]")

print("\nThis explains the remaining divergence!")