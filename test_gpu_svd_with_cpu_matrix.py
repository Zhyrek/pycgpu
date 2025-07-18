#!/usr/bin/env python
"""Test GPU SVD solver with the exact CPU matrix."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

import cupy as cp

# The exact CPU matrix from iteration 2 after consolidation
cpu_matrix = np.array([
    [9.685286e-02, 9.031471e-01, 0.000000e+00],
    [-3.231946e-05, 3.231946e-05, 0.000000e+00],
    [-1.355253e-20, 4.743385e-20, 1.000000e+00]
])

cpu_rhs = np.array([
    -1.991008e+04,
    2.056487e-01,
    -1.204592e-14
])

print("CPU MATRIX TEST IN GPU SVD")
print("=" * 60)
print("\nCPU Matrix (3x3):")
for i in range(3):
    print(f"  Row {i}: ", end="")
    for j in range(3):
        print(f"{cpu_matrix[i,j]:+.10e} ", end="")
    print(f"| RHS: {cpu_rhs[i]:+.10e}")

# Test 1: Use numpy's lstsq (similar to CPU's LAPACK)
print("\n1. NumPy lstsq solution (CPU-like):")
x_numpy, residuals, rank, s = np.linalg.lstsq(cpu_matrix, cpu_rhs, rcond=None)
print(f"   Solution: [{x_numpy[0]:+.15e}, {x_numpy[1]:+.15e}, {x_numpy[2]:+.15e}]")
print(f"   Singular values: {s}")
print(f"   Rank: {rank}")

# Test 2: GPU SVD code
svd_code = '''
extern "C" {

#include <float.h>

__device__ void Singular_Value_Decomposition(double* A, int nrows, int ncols,
                                           double* U, double* s, double* V,
                                           double* superdiagonal);

__device__ void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x);

__global__ void test_svd_solve(double* A, double* B, double* x, 
                               double* singular_values_out, int* rank_out) {
    int tid = threadIdx.x;
    if (tid != 0) return;
    
    // Allocate workspace
    double U[9], V[9], s[3], superdiag[3];
    double A_copy[9], B_copy[3];
    
    // Copy input
    for (int i = 0; i < 9; i++) A_copy[i] = A[i];
    for (int i = 0; i < 3; i++) B_copy[i] = B[i];
    
    // Perform SVD
    Singular_Value_Decomposition(A_copy, 3, 3, U, s, V, superdiag);
    
    // Copy singular values for output
    for (int i = 0; i < 3; i++) singular_values_out[i] = s[i];
    
    // Count effective rank
    double tol = 1e-10 * s[0];
    int rank = 0;
    for (int i = 0; i < 3; i++) {
        if (s[i] > tol) rank++;
    }
    *rank_out = rank;
    
    // Solve the system
    Singular_Value_Decomposition_Solve(U, s, V, 1e-12, 3, 3, B_copy, x);
    
    printf("GPU SVD Debug:\\n");
    printf("  Singular values: %e, %e, %e\\n", s[0], s[1], s[2]);
    printf("  Effective rank: %d\\n", rank);
    printf("  Solution: %+.15e, %+.15e, %+.15e\\n", x[0], x[1], x[2]);
}

''' + open('pycalphad/gpu/svd.c', 'r').read() + '''
}
'''

# Compile and run
module = cp.RawModule(code=svd_code)
test_kernel = module.get_function('test_svd_solve')

# Transfer to GPU
A_gpu = cp.asarray(cpu_matrix.flatten(), dtype=cp.float64)
B_gpu = cp.asarray(cpu_rhs, dtype=cp.float64)
x_gpu = cp.zeros(3, dtype=cp.float64)
s_gpu = cp.zeros(3, dtype=cp.float64)
rank_gpu = cp.zeros(1, dtype=cp.int32)

# Run kernel
test_kernel((1,), (1,), (A_gpu, B_gpu, x_gpu, s_gpu, rank_gpu))
cp.cuda.Stream.null.synchronize()

# Get results
x_result = x_gpu.get()
s_result = s_gpu.get()
rank_result = rank_gpu.get()[0]

print(f"\n2. GPU SVD solution:")
print(f"   Solution: [{x_result[0]:+.15e}, {x_result[1]:+.15e}, {x_result[2]:+.15e}]")
print(f"   Singular values: {s_result}")
print(f"   Rank: {rank_result}")

# Compare solutions
print(f"\n3. Comparison:")
print(f"   Difference in solutions: {np.linalg.norm(x_numpy - x_result):.15e}")
print(f"   Component differences:")
for i in range(3):
    print(f"     x[{i}]: numpy={x_numpy[i]:+.15e}, gpu={x_result[i]:+.15e}, diff={x_numpy[i]-x_result[i]:+.15e}")

# Check what update this would give
print(f"\n4. What this means for convergence:")
print(f"   Delta phase amount: {x_result[0]:+.15e}")
print(f"   Delta chemical potential 0: {x_result[1]:+.15e}")  
print(f"   Delta chemical potential 1: {x_result[2]:+.15e}")

# The phase amount change should adjust X(TI) from 0.9031471 toward 0.9
print(f"\n   Current X(TI) = 0.9031471")
print(f"   Target X(TI) = 0.9")
print(f"   Current error = 0.0031471")
print(f"   If phase amount changes by {x_result[0]:.6e}, this is {'significant' if abs(x_result[0]) > 1e-10 else 'tiny'}")