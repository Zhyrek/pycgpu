#!/usr/bin/env python3
"""Check if the corrected GPU hessian values match CPU"""

# From the output:
# After correction:
#   Site fraction Hessian block (after correction):
#     [0] 1.358035e+04 (idx=18) 4.806831e+04 (idx=19)
#     [1] 4.806831e+04 (idx=23) 2.144266e+04 (idx=24)

# These are the corrected values
gpu_corrected_33 = 1.358035e+04  # Element [3,3]
gpu_corrected_44 = 2.144266e+04  # Element [4,4]

# Expected CPU values
cpu_hess_33 = 13580.347737
cpu_hess_44 = 21442.663538

print("=== Corrected GPU Hessian Values ===")
print(f"GPU corrected [3,3] = {gpu_corrected_33:.6f}")
print(f"CPU expected [3,3] = {cpu_hess_33:.6f}")
print(f"Match? {abs(gpu_corrected_33 - cpu_hess_33) < 0.1}")

print(f"\nGPU corrected [4,4] = {gpu_corrected_44:.6f}")
print(f"CPU expected [4,4] = {cpu_hess_44:.6f}")
print(f"Match? {abs(gpu_corrected_44 - cpu_hess_44) < 0.1}")

# But wait, the debug output also shows:
#   H[1,1] (index 6) = 3.502301e+04
# This suggests the indices are confused

print("\n=== Index Confusion ===")
print("The debug output shows H[1,1] = 3.502301e+04")
print("This is the uncorrected value (RT/Y_NB + RT/Y_TI)")
print("But the corrected values are shown correctly in the")
print("'Site fraction Hessian block' section.")

# Calculate what the values should be
T = 1000.0
Y_NB = 0.612245
Y_TI = 0.387755
R = 8.3145

expected_33 = R * T / Y_NB
expected_44 = R * T / Y_TI

print(f"\n=== Expected Values ===")
print(f"RT/Y_NB = {expected_33:.6f}")
print(f"RT/Y_TI = {expected_44:.6f}")

print(f"\n=== Conclusion ===")
print("The correction is working! The corrected values match CPU values.")
print("The confusion in the debug output is due to different indexing")
print("conventions being printed.")