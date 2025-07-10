#!/usr/bin/env python3
"""Analyze what happens after consolidation"""

print("=== Post-Consolidation Analysis ===")

print("\nThe weighted average of the two phases gives exactly X(TI)=0.4")
print("But CPU iteration 0 shows Y(NB)=0.60315742, not 0.6")

print("\nThis suggests that after consolidation, the CPU:")
print("1. Starts with the weighted average composition (0.6, 0.4)")
print("2. Then runs one Newton step which changes it to (0.60315742, 0.39684258)")

print("\nMeanwhile, the GPU:")
print("1. Keeps the original first phase composition (0.612245, 0.387755)")
print("2. Discards the second phase")

print("\nSo the REAL issue is:")
print("- CPU consolidates phases before equilibrium solver")
print("- GPU does NOT consolidate, just uses first phase")

print("\nThis explains ALL the differences:")
print("1. Different starting compositions")
print("2. Different Hessian values")
print("3. Different c_G values")
print("4. Different final energies")