#!/usr/bin/env python
"""Test matrix inversion differences between CPU and GPU."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Single test condition
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing matrix inversion differences...")
print("="*80)

# Test Hessian for Ti-poor phase
# From CPU output: 
# H[3,3] = 4.428375e+03
# H[3,4] = 1.304530e+04
# H[4,3] = 1.304530e+04
# H[4,4] = 6.790175e+04

hess_block = np.array([[4.428375e+03, 1.304530e+04],
                       [1.304530e+04, 6.790175e+04]])

print("Site fraction Hessian block:")
print(hess_block)
print()

# Invert the matrix
hess_inv = np.linalg.inv(hess_block)

print("Inverted Hessian:")
print(hess_inv)
print()

print("Diagonal elements of inverted Hessian:")
print(f"  [0,0] = {hess_inv[0,0]:.10e}")
print(f"  [1,1] = {hess_inv[1,1]:.10e}")
print()

# Compare with GPU values from debug output
print("GPU full_e_matrix diagonal values from debug output:")
print("  [0,0] = 5.202619e-04")
print("  [1,1] = ? (not shown in output)")
print()

print("Ratio of GPU/CPU diagonal values:")
print(f"  GPU[0,0] / CPU[0,0] = {5.202619e-04 / hess_inv[0,0]}")

# Check if there's a constraint normalization issue
# The CPU diagonal values were: [2.1626519736096025e-05, 2.1626519736096106e-05]
print("\nCPU diagonal values from debug output:")
print("  [0,0] = 2.1626519736096025e-05")
print("  [1,1] = 2.1626519736096106e-05")

# This suggests the CPU is using a different matrix or additional constraints
# Let's check with a constraint
print("\nChecking with constraint...")

# For BCC_A2, the constraint is Y(NB) + Y(TI) = 1
# Constraint Jacobian: [0, 0, 0, 1, 1]
cons_jac = np.array([[1.0, 1.0]])

# Build augmented matrix
n = 2  # phase DOF
m = 1  # number of constraints
augmented = np.zeros((n+m, n+m))
augmented[:n, :n] = hess_block
augmented[n:, :n] = cons_jac
augmented[:n, n:] = cons_jac.T

print("\nAugmented matrix with constraint:")
print(augmented)

# Invert augmented matrix
try:
    aug_inv = np.linalg.inv(augmented)
    print("\nInverted augmented matrix:")
    print(aug_inv)
    print("\nDiagonal of upper-left block:")
    print(f"  [0,0] = {aug_inv[0,0]:.10e}")
    print(f"  [1,1] = {aug_inv[1,1]:.10e}")
except:
    print("\nAugmented matrix is singular!")

print("="*80)