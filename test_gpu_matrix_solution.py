#!/usr/bin/env python
"""Test GPU SVD solver with reconstructed CPU matrix values."""

import numpy as np
import cupy as cp

# From the CPU trace, we know the final iteration has:
# - Single BCC phase after consolidation  
# - X(TI) goes from 0.903147 to exactly 0.900000
# - This is a 3x3 system: [delta_μ_NB, delta_μ_TI, delta_phase_amount]

print("TESTING GPU SVD SOLVER WITH RECONSTRUCTED CPU MATRIX")
print("=" * 60)

# The CPU achieves the constraint by solving a system where:
# The mass balance constraint requires: phase_amount * phase_X_TI = 0.9
# If phase has X(TI)=0.903147, we need phase_amount ≈ 0.9/0.903147 ≈ 0.9965

# From CPU debug output, the constraint violation before final iteration was:
# mass_residual ≈ 0.003147 (since 0.903147 - 0.9 = 0.003147)

print("Reconstructing the CPU's final matrix system...")

# The constraint equation is: sum(phase_amounts * phase_X_TI) = target_X_TI
# For single phase: phase_amount * 0.903147 = 0.9
# So phase_amount should be: 0.9 / 0.903147 = 0.9965

current_phase_amount = 1.0
current_X_TI = 0.903147
target_X_TI = 0.9

mass_residual = current_phase_amount * current_X_TI - target_X_TI
print(f"Current: phase_amount={current_phase_amount}, X(TI)={current_X_TI}")
print(f"Target: X(TI)={target_X_TI}")
print(f"Mass residual: {mass_residual:.6f}")

# The RHS should include this residual
# From GPU trace: component_residual gets subtracted from RHS
# equilibrium_rhs[constraint_row] -= mass_residual

# Approximate the matrix structure based on the constraint:
# Row for mass balance: [0, 0, X_TI_of_phase] * [mu_NB, mu_TI, phase_amount] = target
# This gives us the constraint row

print("\nReconstructing equilibrium matrix...")

# Create a test 3x3 matrix representing the final CPU state
# This is approximate but should capture the key behavior
equilibrium_matrix = np.array([
    [1.0, 0.0, 0.0],      # Chemical potential constraint for NB
    [0.0, 1.0, 0.0],      # Chemical potential constraint for TI  
    [0.0, 0.0, current_X_TI]  # Mass balance: phase_amount * X_TI = target
], dtype=np.float64)

# RHS with the residual subtracted (as CPU does)
equilibrium_rhs = np.array([
    0.0,  # Chemical potential updates (would be small near convergence)
    0.0,  # Chemical potential updates
    target_X_TI - mass_residual  # Mass constraint with residual correction
], dtype=np.float64)

print(f"Matrix:\n{equilibrium_matrix}")
print(f"RHS: {equilibrium_rhs}")

# Solve with NumPy (CPU-like)
cpu_solution = np.linalg.lstsq(equilibrium_matrix, equilibrium_rhs, rcond=1e-16)[0]
print(f"\nCPU solution: {cpu_solution}")

# Now test with our GPU SVD solver
gpu_svd_code = '''
__device__ void svd_solve_3x3(double* A, double* b, double* x) {
    // Simple 3x3 solver for testing
    // A is 3x3 matrix (row-major), b is RHS, x is solution
    
    // For a 3x3 system Ax = b, we can use direct inversion
    double det = A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
    
    if (fabs(det) < 1e-16) {
        // Singular matrix
        x[0] = x[1] = x[2] = 0.0;
        return;
    }
    
    // Cramer's rule for 3x3
    double inv_det = 1.0 / det;
    
    x[0] = inv_det * (b[0]*(A[4]*A[8] - A[5]*A[7]) - b[1]*(A[1]*A[8] - A[2]*A[7]) + b[2]*(A[1]*A[5] - A[2]*A[4]));
    x[1] = inv_det * (b[1]*(A[0]*A[8] - A[2]*A[6]) - b[0]*(A[3]*A[8] - A[5]*A[6]) + b[2]*(A[3]*A[2] - A[0]*A[5]));
    x[2] = inv_det * (b[2]*(A[0]*A[4] - A[1]*A[3]) - b[0]*(A[6]*A[4] - A[7]*A[3]) + b[1]*(A[6]*A[1] - A[7]*A[0]));
}

extern "C" __global__ void test_gpu_solver(double* matrix, double* rhs, double* solution) {
    svd_solve_3x3(matrix, rhs, solution);
}
'''

# Compile and run GPU solver
try:
    gpu_module = cp.RawModule(code=gpu_svd_code)
    gpu_kernel = gpu_module.get_function('test_gpu_solver')
    
    # Copy data to GPU
    d_matrix = cp.asarray(equilibrium_matrix.flatten(), dtype=cp.float64)
    d_rhs = cp.asarray(equilibrium_rhs, dtype=cp.float64)
    d_solution = cp.zeros(3, dtype=cp.float64)
    
    # Run GPU solver
    gpu_kernel((1,), (1,), (d_matrix, d_rhs, d_solution))
    
    gpu_solution = d_solution.get()
    print(f"GPU solution: {gpu_solution}")
    
    # Compare solutions
    diff = np.abs(cpu_solution - gpu_solution)
    print(f"\nDifference: {diff}")
    print(f"Max difference: {np.max(diff):.2e}")
    
    if np.max(diff) < 1e-12:
        print("✓ GPU and CPU solvers agree!")
    else:
        print("✗ GPU and CPU solvers disagree!")
        
except Exception as e:
    print(f"GPU test failed: {e}")
    print("Testing with NumPy only...")

# Test the key insight: what update does this produce?
print(f"\nAnalyzing the solution:")
print(f"Chemical potential updates: μ_NB={cpu_solution[0]:.6f}, μ_TI={cpu_solution[1]:.6f}")
print(f"Phase amount update: Δ_phase={cpu_solution[2]:.6f}")

new_phase_amount = current_phase_amount + cpu_solution[2]
new_overall_X_TI = new_phase_amount * current_X_TI

print(f"\nAfter applying update:")
print(f"New phase amount: {new_phase_amount:.6f}")
print(f"New overall X(TI): {new_overall_X_TI:.6f}")
print(f"Target X(TI): {target_X_TI:.6f}")
print(f"Error: {abs(new_overall_X_TI - target_X_TI):.6f}")