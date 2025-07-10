#!/usr/bin/env python3
"""Check the structure of GPU hessian to understand the spurious terms better"""

import numpy as np

# GPU hessian values from output
gpu_hess_orig = np.array([[35023, 48068],
                          [48068, 35023]])

# CPU hessian values
cpu_hess = np.array([[13857, 13045],
                     [13045, 20786]])

print("=== Hessian Structure Analysis ===")
print(f"GPU original diagonal: {gpu_hess_orig[0,0]:.0f}, {gpu_hess_orig[1,1]:.0f}")
print(f"CPU diagonal: {cpu_hess[0,0]:.0f}, {cpu_hess[1,1]:.0f}")
print(f"Diagonal ratios: {gpu_hess_orig[0,0]/cpu_hess[0,0]:.3f}, {gpu_hess_orig[1,1]/cpu_hess[1,1]:.3f}")

print(f"\nGPU off-diagonal: {gpu_hess_orig[0,1]:.0f}")
print(f"CPU off-diagonal: {cpu_hess[0,1]:.0f}")
print(f"Off-diagonal ratio: {gpu_hess_orig[0,1]/cpu_hess[0,1]:.3f}")

# The ideal mixing contribution to hessian for binary system
# H[i,j] = d²G/dY_i dY_j
# For ideal mixing: G_ideal = RT*(Y1*ln(Y1) + Y2*ln(Y2))
# d²G/dY1² = RT/Y1
# d²G/dY1 dY2 = 0 (no cross terms in ideal mixing)

# But with the spurious /(Y1+Y2) factor:
# G_spurious = RT*(Y1*ln(Y1) + Y2*ln(Y2))/(Y1+Y2)
# This creates cross-terms

R = 8.3145
T = 1000
Y1 = 0.6
Y2 = 0.4

# Correct hessian (CPU)
H11_correct = R * T / Y1
H22_correct = R * T / Y2
H12_correct = 0  # No cross terms in ideal mixing

print(f"\n=== Theoretical Values ===")
print(f"Correct H[1,1] = RT/Y1 = {H11_correct:.0f}")
print(f"Correct H[2,2] = RT/Y2 = {H22_correct:.0f}")
print(f"Correct H[1,2] = 0")

# With spurious /(Y1+Y2) factor, the hessian becomes more complex
# Need to differentiate: RT*(Y1*ln(Y1) + Y2*ln(Y2))/(Y1+Y2)
print(f"\n=== Understanding GPU Spurious Terms ===")

# The GPU includes both RT/Y1 and RT/Y2 in diagonal elements
H11_gpu_spurious = R * T / Y1 + R * T / Y2
H22_gpu_spurious = R * T / Y1 + R * T / Y2

print(f"GPU spurious H[1,1] = RT/Y1 + RT/Y2 = {H11_gpu_spurious:.0f}")
print(f"Ratio to CPU: {H11_gpu_spurious/H11_correct:.3f}")

# But this doesn't match the observed 2.53 ratio exactly
# The actual GPU hessian must have additional terms from the complex differentiation

# Check if subtracting RT/Y2 from H[1,1] gives the right value
H11_corrected = gpu_hess_orig[0,0] - R * T / Y2
print(f"\nCorrected H[1,1] = {gpu_hess_orig[0,0]} - {R*T/Y2:.0f} = {H11_corrected:.0f}")
print(f"CPU H[1,1] = {cpu_hess[0,0]:.0f}")
print(f"Match? {abs(H11_corrected - cpu_hess[0,0]) < 100}")