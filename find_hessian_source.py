#!/usr/bin/env python3
"""
Search for where the specific Hessian values come from.
hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04
"""

import os
import re
import subprocess

# These values might be in scientific notation or regular form
patterns = [
    r'5\.543.*e\+03',
    r'5543\.0',
    r'4\.9887.*e\+04', 
    r'49887\.0',
    r'hess\[3,3\].*5\.543',
    r'hess\[4,4\].*4\.9887',
]

print("Searching for Hessian values: hess[3,3] = 5.543000e+03 and hess[4,4] = 4.988700e+04")
print("="*80)

# Search in all Python files
print("\nSearching Python files...")
for pattern in patterns:
    cmd = f'grep -r -n "{pattern}" --include="*.py" . 2>/dev/null | head -5'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"\nPattern '{pattern}' found in Python files:")
        print(result.stdout)

# Search in all C/CUDA files
print("\nSearching C/CUDA files...")
for pattern in patterns:
    cmd = f'grep -r -n "{pattern}" --include="*.c" --include="*.cu" --include="*.h" . 2>/dev/null | head -5'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"\nPattern '{pattern}' found in C/CUDA files:")
        print(result.stdout)

# Search in markdown/documentation
print("\nSearching documentation...")
for pattern in patterns:
    cmd = f'grep -r -n "{pattern}" --include="*.md" --include="*.txt" . 2>/dev/null | head -5'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"\nPattern '{pattern}' found in documentation:")
        print(result.stdout)

# Check if these might be calculated values at specific conditions
print("\n" + "="*80)
print("Checking if these are calculated values...")

# BCC_A2 has 2 site fractions, so Hessian would be 2x2 for site fractions
# But if we include state variables (T, P, N), it could be 5x5
# hess[3,3] and hess[4,4] would be the site fraction diagonal elements

print("\nFor BCC_A2 phase with DOF ordering [T, P, N, Y_NB, Y_TI]:")
print("- hess[3,3] would be d²G/dY_NB²")
print("- hess[4,4] would be d²G/dY_TI²")
print("\nThese values likely come from:")
print("- Temperature = 300K or 1000K")
print("- Specific composition Y(NB)=0.773706, Y(TI)=0.226294")
print("- Or Y(NB)=0.7, Y(TI)=0.3")

# Try to calculate what these values might be
R = 8.314462618  # J/(mol*K)
T_values = [300, 1000]
y_values = [(0.773706, 0.226294), (0.7, 0.3)]

print("\nPossible RT/y values:")
for T in T_values:
    for y_nb, y_ti in y_values:
        rt_y_nb = R * T / y_nb
        rt_y_ti = R * T / y_ti
        print(f"  T={T}K, Y(NB)={y_nb:.6f}: RT/Y(NB) = {rt_y_nb:.6e}")
        print(f"  T={T}K, Y(TI)={y_ti:.6f}: RT/Y(TI) = {rt_y_ti:.6e}")
        
        # Check if these match our target values
        if abs(rt_y_nb - 5543.0) < 100:
            print("    *** Possible match for hess[3,3] = 5.543e+03! ***")
        if abs(rt_y_ti - 49887.0) < 1000:
            print("    *** Possible match for hess[4,4] = 4.9887e+04! ***")