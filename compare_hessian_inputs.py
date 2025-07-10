#!/usr/bin/env python3
"""Compare Hessian calculation inputs"""

print("=== Comparing Hessian Inputs ===")

print("\nFrom debug outputs:")
print("\nCPU Phase 0 (iteration 0):")
print("  Site fractions: [0.60315742 0.39684258]")
print("  Workspace DOF: [1.0, 101325.0, 1000.0, 0.60315742, 0.39684258]")

print("\nGPU Phase 0 (iteration 0):")  
print("  workspace_dof values: [1.000000, 101325.000000, 1000.000000, 0.612245, 0.387755]")
print("  Site fractions: [0.612245, 0.387755]")

print("\nThe site fractions are DIFFERENT!")
print("CPU: Y(NB)=0.60315742, Y(TI)=0.39684258")
print("GPU: Y(NB)=0.612245,   Y(TI)=0.387755")

print("\nDifferences:")
print("  ΔY(NB) = 0.612245 - 0.60315742 = 0.00908758")
print("  ΔY(TI) = 0.387755 - 0.39684258 = -0.00908758")

print("\nThis explains why the Hessian values are different!")
print("The GPU is using the INITIAL site fractions from the hull calculation,")
print("while the CPU has already updated them during phase consolidation.")