#!/usr/bin/env python3
"""Trace GPU phase matrix calculation"""

print("=== GPU Phase Matrix Analysis ===")

print("\nFrom GPU debug output, the phase matrix construction shows:")
print("  Site fraction Hessian block:")
print("    [0] 3.502301e+04 (idx=18) 4.806831e+04 (idx=19)")
print("    [1] 4.806831e+04 (idx=23) 3.502301e+04 (idx=24)")

print("\nBut the CPU shows:")
print("  hess[3,3] = 1.358035e+04")
print("  hess[3,4] = 1.304530e+04")

print("\nThe GPU Hessian values are completely different!")
print("GPU: 3.502301e+04, CPU: 1.358035e+04")
print("Ratio: 3.502301e+04 / 1.358035e+04 = 2.58")

print("\nThis explains why the c_G values are different!")
print("The GPU is using a different Hessian matrix.")

print("\nPossible causes:")
print("1. Different site fraction values when computing Hessian")
print("2. Different Hessian calculation formula")
print("3. Wrong indices when extracting Hessian values")