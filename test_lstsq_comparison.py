#!/usr/bin/env python3
"""Direct comparison of CPU LAPACK lstsq vs GPU SVD solver"""
import numpy as np
import cupy as cp
from pycalphad.core.minimizer import lstsq_check_infeasible

# Test the SVD solver directly
svd_code = """
extern "C" {

// Include the SVD solver code
#include <math.h>
#include <float.h>

#define MAX_SVD_DIM 10

// Declare the SVD functions (implementations would be linked from svd.c)
__device__ int Singular_Value_Decomposition(double* A, int nrows, int ncols, double* U, 
                      double* singular_values, double* V, double* dummy_array);
__device__ void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x);

__global__ void test_lstsq_kernel(double* A, int nrows, int ncols, double* b, double* x_out) {
    // Storage for SVD
    double U[MAX_SVD_DIM * MAX_SVD_DIM];
    double V[MAX_SVD_DIM * MAX_SVD_DIM];
    double singular_values[MAX_SVD_DIM];
    double superdiag[MAX_SVD_DIM];
    
    // Make a copy of A since SVD modifies it
    double A_copy[MAX_SVD_DIM * MAX_SVD_DIM];
    for (int i = 0; i < nrows * ncols; ++i) {
        A_copy[i] = A[i];
    }
    
    // Make a copy of b since we'll modify it
    double b_copy[MAX_SVD_DIM];
    for (int i = 0; i < nrows; ++i) {
        b_copy[i] = b[i];
    }
    
    // Perform SVD
    int svd_result = Singular_Value_Decomposition(A_copy, nrows, ncols, U, singular_values, V, superdiag);
    
    if (svd_result != 0) {
        printf("SVD failed\\n");
        for (int i = 0; i < ncols; ++i) {
            x_out[i] = 0.0;
        }
        return;
    }
    
    // Print singular values
    printf("GPU SVD singular values: ");
    for (int i = 0; i < ncols; ++i) {
        printf("%.6e ", singular_values[i]);
    }
    printf("\\n");
    
    // Solve using SVD
    double x_temp[MAX_SVD_DIM];
    double tolerance = 1e-16;
    Singular_Value_Decomposition_Solve(U, singular_values, V, tolerance, nrows, ncols, b_copy, x_temp);
    
    // Copy result
    for (int i = 0; i < ncols; ++i) {
        x_out[i] = x_temp[i];
    }
    
    // Check residual
    double residual = 0.0;
    for (int i = 0; i < nrows; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < ncols; ++j) {
            row_sum += A[i * ncols + j] * x_temp[j];
        }
        double diff = row_sum - b[i];
        residual += diff * diff;
    }
    printf("GPU residual: %.15e\\n", residual);
}

}
"""

# Test cases
print("=== Test 1: Well-conditioned 3x3 system ===")
A1 = np.array([[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 10.0]], dtype=np.float64)
b1 = np.array([6.0, 15.0, 25.0], dtype=np.float64)

# CPU solution using NumPy (which uses LAPACK)
x_cpu1, residuals, rank, s = np.linalg.lstsq(A1, b1, rcond=1e-16)
print(f"CPU solution: {x_cpu1}")
print(f"CPU singular values: {s}")
print(f"CPU rank: {rank}")
print(f"CPU residual: {np.sum((A1 @ x_cpu1 - b1)**2):.15e}")

# CPU solution using pycalphad's lstsq_check_infeasible
x_pycalphad1 = np.zeros(3, dtype=np.float64)
lstsq_check_infeasible(A1.copy(), b1.copy(), x_pycalphad1)
print(f"Pycalphad CPU solution: {x_pycalphad1}")

print("\n=== Test 2: Rank-deficient 3x3 system (infeasible constraint) ===")
# This mimics the infeasible system from the GPU test
# One phase row, one mole fraction constraint, one system amount constraint
A2 = np.array([[0.6031574, 0.3968426, 0.0],          # Phase row
               [-0.0001157, 0.0001157, 0.0],         # Mole fraction constraint
               [0.0, 0.0, 1.0]], dtype=np.float64)    # System amount constraint
b2 = np.array([-49777.33, 0.5146, 0.0], dtype=np.float64)

# CPU solution
x_cpu2, residuals2, rank2, s2 = np.linalg.lstsq(A2, b2, rcond=1e-16)
print(f"CPU solution: {x_cpu2}")
print(f"CPU singular values: {s2}")
print(f"CPU rank: {rank2}")
print(f"CPU residual: {np.sum((A2 @ x_cpu2 - b2)**2):.15e}")

# Pycalphad CPU
x_pycalphad2 = np.zeros(3, dtype=np.float64)
lstsq_check_infeasible(A2.copy(), b2.copy(), x_pycalphad2)
print(f"Pycalphad CPU solution: {x_pycalphad2}")
if np.any(np.isnan(x_pycalphad2)):
    print("Pycalphad detected infeasible system (returned NaN)")

print("\n=== Test 3: Actual infeasible system from equilibrium solver ===")
# This is from iteration 1 of the GPU debug output
A3 = np.array([[0.1616249, 0.8383751, 0.0],
               [4.452274e-5, -8.904548e-5, -7.232520e-17],
               [3.803688e-5, -7.607376e-5, 1.0]], dtype=np.float64)
b3 = np.array([-46924.18, -0.8867288, -0.3754448], dtype=np.float64)

# CPU solution
x_cpu3, residuals3, rank3, s3 = np.linalg.lstsq(A3, b3, rcond=1e-16)
print(f"CPU solution: {x_cpu3}")
print(f"CPU singular values: {s3}")
print(f"CPU rank: {rank3}")
print(f"CPU residual: {np.sum((A3 @ x_cpu3 - b3)**2):.15e}")
print(f"Phase amount change (x[2]): {x_cpu3[2]:.6f}")

# Check what happens with the system amount
print(f"System amount after applying solution: {-0.3754448 + x_cpu3[2]:.6f}")

# Pycalphad CPU
x_pycalphad3 = np.zeros(3, dtype=np.float64)
lstsq_check_infeasible(A3.copy(), b3.copy(), x_pycalphad3)
print(f"Pycalphad CPU solution: {x_pycalphad3}")
if np.any(np.isnan(x_pycalphad3)):
    print("Pycalphad detected infeasible system (returned NaN)")

# Now test with GPU SVD solver (if available)
try:
    # Load the actual SVD implementation
    svd_impl = open('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/svd.c', 'r').read()
    
    # Create a complete kernel with SVD implementation
    full_kernel = svd_impl + svd_code
    
    # Compile
    mod = cp.RawModule(code=full_kernel, backend='nvcc')
    test_kernel = mod.get_function('test_lstsq_kernel')
    
    print("\n=== GPU SVD Solver Test ===")
    
    # Test case 3 on GPU
    A_gpu = cp.asarray(A3.flatten(), dtype=cp.float64)
    b_gpu = cp.asarray(b3, dtype=cp.float64)
    x_gpu = cp.zeros(3, dtype=cp.float64)
    
    print("Testing infeasible system on GPU...")
    test_kernel((1,), (1,), (A_gpu, 3, 3, b_gpu, x_gpu))
    
    x_gpu_result = x_gpu.get()
    print(f"GPU solution: {x_gpu_result}")
    print(f"GPU phase amount change (x[2]): {x_gpu_result[2]:.6f}")
    
    # Compare
    print(f"\nCPU vs GPU difference in solution: {np.linalg.norm(x_cpu3 - x_gpu_result):.6e}")
    print(f"CPU phase change: {x_cpu3[2]:.6f}, GPU phase change: {x_gpu_result[2]:.6f}")
    
except Exception as e:
    print(f"\nGPU test skipped: {e}")
    print("This is expected if not running on a GPU system")