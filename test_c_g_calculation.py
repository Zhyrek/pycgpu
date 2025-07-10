#!/usr/bin/env python3
"""Test c_G calculation logic"""
import numpy as np

print("=== Understanding c_G Calculation ===")

# From debug output
print("\nPhase 0 values:")
print("full_e_matrix (2x2):")
print("  [0,0] = -3.832798e-05")
print("  [0,1] = 3.832798e-05")
print("  [1,0] = 3.832798e-05")
print("  [1,1] = -3.832798e-05")

print("\nGradient (starting from num_statevars=3):")
print("  grad[3] = -4.008941e+04")
print("  grad[4] = -3.574718e+04")

# CPU calculation
e_matrix = np.array([[-3.832798e-05, 3.832798e-05],
                     [3.832798e-05, -3.832798e-05]])
grad = np.array([-4.008941e+04, -3.574718e+04])

print("\nCPU calculation (c_G = -e_matrix @ grad):")
c_G_cpu = -e_matrix @ grad
print(f"  c_G[0] = {c_G_cpu[0]:.6f}")
print(f"  c_G[1] = {c_G_cpu[1]:.6f}")

print("\nGPU calculation (as shown in debug):")
c_G_gpu = np.zeros(2)
for i in range(2):
    for j in range(2):
        c_G_gpu[i] -= e_matrix[i,j] * grad[j]
        print(f"  c_G[{i}] -= e_matrix[{i},{j}] * grad[{j}] = {e_matrix[i,j]} * {grad[j]} = {e_matrix[i,j] * grad[j]}")

print(f"\nGPU result:")
print(f"  c_G[0] = {c_G_gpu[0]:.6f}")
print(f"  c_G[1] = {c_G_gpu[1]:.6f}")

print("\nThese should be the same! Let me check the actual CPU debug values...")
print("\nFrom CPU debug: c_G values: [ 0.48612027 -0.48612027]")
print("From GPU debug: c_G[0] = -1.664287e-01, c_G[1] = 1.664287e-01")

print("\nWait, let me recalculate with the exact values from the debug output...")
# The issue might be in the e_matrix values!