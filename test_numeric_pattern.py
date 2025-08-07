#!/usr/bin/env python
"""Analyze the numeric pattern in the errors."""

# Failing conditions
# Condition 10: GPU=-26853.831051, CPU=-27185.304121, Diff=331.473071
# Condition 17: GPU=-34593.928904, CPU=-33366.599950, Diff=1227.328955

print("Numeric pattern analysis:")
print("="*60)

# Error values
error_10 = 331.473071
error_17 = 1227.328955

print(f"Error for condition 10: {error_10}")
print(f"Error for condition 17: {error_17}")
print(f"Ratio: {error_17 / error_10:.6f}")

# Check if errors relate to thread IDs or modulo 7
print("\nRelationship to thread IDs:")
print(f"Error 10 / thread 10 = {error_10 / 10:.6f}")
print(f"Error 17 / thread 17 = {error_17 / 17:.6f}")

print("\nRelationship to modulo 7:")
print(f"Error 10 / 3 = {error_10 / 3:.6f}")  # thread 10 % 7 = 3
print(f"Error 17 / 3 = {error_17 / 3:.6f}")  # thread 17 % 7 = 3

# Check GPU values
gpu_10 = -26853.831051
gpu_17 = -34593.928904
cpu_10 = -27185.304121
cpu_17 = -33366.599950

print("\nSign of errors:")
print(f"Condition 10: GPU > CPU by {error_10} (GPU less negative)")
print(f"Condition 17: GPU < CPU by {error_17} (GPU more negative)")

# Check if there's a pattern in the absolute values
print("\nAbsolute value analysis:")
print(f"|GPU_10| = {abs(gpu_10):.6f}")
print(f"|CPU_10| = {abs(cpu_10):.6f}")
print(f"|GPU_17| = {abs(gpu_17):.6f}")
print(f"|CPU_17| = {abs(cpu_17):.6f}")

# Phase amounts from earlier tests
print("\nPhase amounts:")
print("Condition 10: FCC_A1=0.100001, AU2BI_C15=0.899999")
print("Condition 17: FCC_A1=0.400024, AU2BI_C15=0.599976")

# Check if error relates to phase amounts
fcc_10 = 0.100001
au2bi_10 = 0.899999
fcc_17 = 0.400024
au2bi_17 = 0.599976

print("\nError per FCC_A1 amount:")
print(f"Condition 10: {error_10 / fcc_10:.2f}")
print(f"Condition 17: {error_17 / fcc_17:.2f}")

print("\nError per AU2BI_C15 amount:")
print(f"Condition 10: {error_10 / au2bi_10:.2f}")
print(f"Condition 17: {error_17 / au2bi_17:.2f}")

# Temperature dependence?
print("\nTemperature analysis:")
print("Condition 10: T=500K")
print("Condition 17: T=600K")
print(f"Error 10 / 500 = {error_10 / 500:.6f}")
print(f"Error 17 / 600 = {error_17 / 600:.6f}")