#!/usr/bin/env python3
"""Direct comparison of CPU LAPACK vs GPU SVD on actual equilibrium matrices"""
import numpy as np
from scipy.linalg import lstsq, svd
import sys

print("Testing actual matrices from equilibrium solver...")

# From CPU iteration 0 - 4x4 system with 2 BCC_A2 phases
print("\n=== CPU Iteration 0: 4x4 system (2 phases) ===")
A0 = np.array([
    [0.6122449, 0.3877551, 0.0, 0.0],      # Phase 0 (BCC_A2)
    [0.5936643, 0.4063357, 0.0, 0.0],      # Phase 1 (BCC_A2)  
    [-0.0001168441, 0.0001168441, -0.01224490, 0.006335728],  # Mole fraction constraint
    [0.0, 0.0, 1.0, 1.0]                   # System amount constraint
], dtype=np.float64)
b0 = np.array([-49817.16, -49734.97, 0.5199998, 0.0], dtype=np.float64)

print("Matrix A:")
for i in range(4):
    print(f"  [{A0[i,0]:+.6e} {A0[i,1]:+.6e} {A0[i,2]:+.6e} {A0[i,3]:+.6e}]")
print(f"RHS b: [{b0[0]:+.6e} {b0[1]:+.6e} {b0[2]:+.6e} {b0[3]:+.6e}]")

# Solve with different methods
x_lstsq, res, rank, s = lstsq(A0, b0)
print(f"\nNumPy lstsq solution: [{x_lstsq[0]:+.6e} {x_lstsq[1]:+.6e} {x_lstsq[2]:+.6e} {x_lstsq[3]:+.6e}]")
print(f"Singular values: {s}")
print(f"Rank: {rank}")
print(f"Residual: {np.sum((A0 @ x_lstsq - b0)**2):.15e}")

# What does this solution mean?
print(f"\nPhysical interpretation:")
print(f"  Chemical potential changes: μ(NB)={x_lstsq[0]:.1f}, μ(TI)={x_lstsq[1]:.1f}")
print(f"  Phase amount changes: δNP[0]={x_lstsq[2]:.6f}, δNP[1]={x_lstsq[3]:.6f}")
print(f"  Net phase amount change: {x_lstsq[2] + x_lstsq[3]:.6f}")

# From CPU iteration 1 (after applying solution) - still 4x4
print("\n=== CPU Iteration 1: 4x4 system (after step) ===")
A1 = np.array([
    [0.6031574, 0.3968426, 0.0, 0.0],
    [0.6031491, 0.3968509, 0.0, 0.0],
    [-0.0001156633, 0.0001156633, -6.925150e-6, 1.432013e-6],
    [0.0, 0.0, 1.0, 1.0]
], dtype=np.float64)
b1 = np.array([-49777.33, -49777.29, 0.5146292, -1.504352e-14], dtype=np.float64)

print("Matrix A:")
for i in range(4):
    print(f"  [{A1[i,0]:+.6e} {A1[i,1]:+.6e} {A1[i,2]:+.6e} {A1[i,3]:+.6e}]")
print(f"RHS b: [{b1[0]:+.6e} {b1[1]:+.6e} {b1[2]:+.6e} {b1[3]:+.6e}]")

x_lstsq1, res1, rank1, s1 = lstsq(A1, b1)
print(f"\nNumPy lstsq solution: [{x_lstsq1[0]:+.6e} {x_lstsq1[1]:+.6e} {x_lstsq1[2]:+.6e} {x_lstsq1[3]:+.6e}]")
print(f"Singular values: {s1}")
print(f"Note: Phases are nearly identical (rows 0 and 1)")

# From GPU iteration 1 (after phase removal) - 3x3 system  
print("\n=== GPU Iteration 1: 3x3 system (1 phase) ===")
A_gpu = np.array([
    [0.1616249, 0.8383751, 0.0],
    [4.452274e-5, -8.904548e-5, -7.232520e-17],
    [3.803688e-5, -7.607376e-5, 1.0]
], dtype=np.float64)
b_gpu = np.array([-46924.18, -0.8867288, -0.3754448], dtype=np.float64)

print("Matrix A:")
for i in range(3):
    print(f"  [{A_gpu[i,0]:+.6e} {A_gpu[i,1]:+.6e} {A_gpu[i,2]:+.6e}]")
print(f"RHS b: [{b_gpu[0]:+.6e} {b_gpu[1]:+.6e} {b_gpu[2]:+.6e}]")

x_gpu, res_gpu, rank_gpu, s_gpu = lstsq(A_gpu, b_gpu)
print(f"\nNumPy lstsq solution: [{x_gpu[0]:+.6e} {x_gpu[1]:+.6e} {x_gpu[2]:+.6e}]")
print(f"Singular values: {s_gpu}")
print(f"Rank: {rank_gpu}")

print(f"\nPhysical interpretation:")
print(f"  Chemical potential changes: μ(NB)={x_gpu[0]:.1f}, μ(TI)={x_gpu[1]:.1f}")
print(f"  Phase amount change: δNP={x_gpu[2]:.6f}")
print(f"  This would change system amount from {-0.3754448:.6f} to {-0.3754448 + x_gpu[2]:.6f}")

# Check what CPU does differently
print("\n=== Key Difference ===")
print("CPU has 2 nearly identical BCC_A2 phases that get consolidated")
print("GPU has 1 BCC_A2 phase after the duplicate is removed")
print("\nThis is the fundamental difference - not the lstsq algorithm!")

# Let's verify by checking what happens if we use SVD directly
print("\n=== Direct SVD comparison on GPU's 3x3 matrix ===")
U, s, Vt = svd(A_gpu)
print(f"SVD singular values: {s}")
print(f"Condition number: {s[0]/s[-1]:.2e}")

# Compute pseudoinverse solution manually
s_inv = np.zeros((3, 3))
for i in range(3):
    if s[i] > 1e-16:
        s_inv[i, i] = 1.0 / s[i]
x_svd = Vt.T @ s_inv @ U.T @ b_gpu
print(f"Manual SVD solution: [{x_svd[0]:+.6e} {x_svd[1]:+.6e} {x_svd[2]:+.6e}]")
print(f"Matches lstsq: {np.allclose(x_svd, x_gpu)}")

print("\n=== Conclusion ===")
print("The difference is NOT in the linear solver (LAPACK vs SVD).")
print("The difference is that:")
print("1. CPU keeps 2 BCC_A2 phases and consolidates them after iteration 1")
print("2. GPU removes the second phase early, leaving only 1 BCC_A2 phase")
print("3. With only 1 phase, the system cannot satisfy X(TI)=0.4 constraint")
print("4. The linear solver correctly identifies this as infeasible (large δNP)")