#!/usr/bin/env python3
"""Analyze the difference in hessian calculation between CPU and GPU"""

import numpy as np

# From the symbolic analysis, we know:
# - CPU evaluates d²G/dY_NB² = 1.385750e+04 at T=1000, Y_NB=0.6, Y_TI=0.4
# - GPU evaluates to ~3.50e+04 at the same point

# The GPU generated code for out[18] (H[3,3]) starts with:
# -2.0*(x[3]*(...) + x[4]*(...))/(x[3] + x[4])^2 + 26090.6*x[4]/(x[3] + x[4]) + ...

# Let's analyze the structure of the hessian

def analyze_hessian_structure():
    """Analyze the structure of d²G/dY_NB² for G = (Y_NB + Y_TI) * g(Y_NB, Y_TI, T)"""
    
    print("=== Hessian Structure Analysis ===")
    print("\nFor G = (Y_NB + Y_TI) * g(Y_NB, Y_TI, T), where g is the core energy function")
    print("\nFirst derivative:")
    print("dG/dY_NB = g + (Y_NB + Y_TI) * dg/dY_NB")
    print("\nSecond derivative:")
    print("d²G/dY_NB² = dg/dY_NB + dg/dY_NB + (Y_NB + Y_TI) * d²g/dY_NB²")
    print("         = 2 * dg/dY_NB + (Y_NB + Y_TI) * d²g/dY_NB²")
    
    print("\nFor a binary system where Y_NB + Y_TI = 1:")
    print("d²G/dY_NB² = 2 * dg/dY_NB + d²g/dY_NB²")
    
    print("\n=== Hypothesis ===")
    print("The GPU might be including additional terms or not simplifying (Y_NB + Y_TI) = 1")
    
    # Let's check the interaction parameter
    print("\n=== Interaction Parameter ===")
    print("From the code, we see: 26090.6*x[4]/(x[3] + x[4])")
    print("This is likely: L_NB,TI * Y_TI / (Y_NB + Y_TI)")
    print("For Y_NB=0.6, Y_TI=0.4, this gives: 26090.6 * 0.4 / 1.0 = 10436.24")
    
    # Check if this matches the numerical difference
    print("\n=== Checking Scaling ===")
    cpu_hess = 13857.50  # approximate CPU value
    gpu_hess = 35023.01  # approximate GPU value
    ratio = gpu_hess / cpu_hess
    print(f"GPU/CPU ratio: {ratio:.3f}")
    
    # The BCC_A2 phase for NbTi likely has two sublattices
    print("\n=== Possible Issue ===")
    print("The GPU code might be handling the sublattice model differently")
    print("For a two-sublattice model, there might be additional scaling factors")
    
analyze_hessian_structure()

# Let's also check if there's a pattern in the generated code
print("\n=== Code Pattern Analysis ===")
print("The GPU hessian code has terms like:")
print("1. Energy terms divided by (x[3] + x[4])^2")
print("2. Interaction terms like 26090.6*x[4]/(x[3] + x[4])")
print("3. Additional terms with (x[3] + x[4]) factors")
print("\nThis suggests the (Y_NB + Y_TI) factor is not being simplified to 1.0")