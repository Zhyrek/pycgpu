#!/usr/bin/env python
"""Check the condition number of the problematic matrix."""

import numpy as np

def main():
    """Check matrix condition."""
    
    print("=" * 80)
    print("MATRIX CONDITION ANALYSIS")
    print("=" * 80)
    
    # The GPU iteration 1 matrix that causes the huge jump
    A = np.array([
        [2.513799e-01, 2.944523e-13, 7.486201e-01, 0.0, 0.0, 0.0],
        [1.529026e-01, 8.470974e-01, 2.669857e-13, 0.0, 0.0, 0.0],
        [1.529007e-01, 8.470993e-01, 2.669863e-13, 0.0, 0.0, 0.0],
        [4.042178e-06, -1.926709e-06, -2.115469e-06, 5.882316e-02, -3.965420e-02, -3.965610e-02],
        [-1.926709e-06, 1.926709e-06, -4.217165e-17, -5.059939e-01, 3.411035e-01, 3.411054e-01],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    ])
    
    b = np.array([-5.600096e+04, -5.246191e+04, -5.191523e+04, 
                  -2.107682e-01, 9.583567e-02, -3.685940e-14])
    
    print("\nMatrix A:")
    print(A)
    
    print("\nRHS b:")
    print(b)
    
    # Check condition number
    cond = np.linalg.cond(A)
    print(f"\nCondition number: {cond:.3e}")
    
    if cond > 1e10:
        print("⚠ Matrix is ILL-CONDITIONED (condition number > 1e10)")
    elif cond > 1e6:
        print("⚠ Matrix is poorly conditioned (condition number > 1e6)")
    else:
        print("✓ Matrix conditioning is acceptable")
    
    # Check determinant
    det = np.linalg.det(A)
    print(f"\nDeterminant: {det:.3e}")
    
    if abs(det) < 1e-10:
        print("⚠ Matrix is nearly SINGULAR (determinant ≈ 0)")
    
    # Solve using numpy (which uses LAPACK like CPU)
    try:
        x_numpy = np.linalg.solve(A, b)
        print("\nNumPy solution (like CPU):")
        print(f"  x = {x_numpy}")
        
        # Check residual
        residual = np.linalg.norm(A @ x_numpy - b)
        print(f"  Residual: {residual:.3e}")
    except np.linalg.LinAlgError as e:
        print(f"\nNumPy failed to solve: {e}")
    
    # Check using least squares (more stable)
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("\nLeast squares solution:")
    print(f"  x = {x_lstsq}")
    print(f"  Rank: {rank} (should be 6 for full rank)")
    print(f"  Singular values: {s}")
    
    # Check for near-duplicate rows
    print("\n" + "-" * 50)
    print("CHECKING FOR NEAR-DUPLICATE ROWS:")
    print("-" * 50)
    
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            # Normalize rows for comparison
            row_i = A[i] / (np.linalg.norm(A[i]) + 1e-15)
            row_j = A[j] / (np.linalg.norm(A[j]) + 1e-15)
            
            diff = np.linalg.norm(row_i - row_j)
            if diff < 1e-6:
                print(f"⚠ Rows {i} and {j} are nearly identical! (diff={diff:.3e})")
                print(f"  Row {i}: {A[i]}")
                print(f"  Row {j}: {A[j]}")
    
    # Check what happens with slight perturbation
    print("\n" + "-" * 50)
    print("PERTURBATION ANALYSIS:")
    print("-" * 50)
    
    # Add tiny perturbation to see if it stabilizes
    A_perturbed = A.copy()
    A_perturbed[2, 1] += 1e-6  # Slightly perturb the duplicate row
    
    x_perturbed = np.linalg.solve(A_perturbed, b)
    print(f"\nWith tiny perturbation to row 2:")
    print(f"  x = {x_perturbed}")
    
    # Compare to expected GPU bad solution
    gpu_bad = np.array([-2.44e+08, 4.41e+07, 8.20e+07, -9.22e+02, 7.05e+08, -7.05e+08])
    print(f"\nGPU bad solution:")
    print(f"  x = {gpu_bad}")
    
    # Check if GPU solution satisfies the equations
    gpu_residual = A @ gpu_bad - b
    print(f"\nGPU solution residual: {np.linalg.norm(gpu_residual):.3e}")
    print(f"  Individual residuals: {gpu_residual}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The matrix has near-duplicate rows which makes it ill-conditioned.")
    print("The GPU LU solver produces wildly incorrect results due to this.")
    print("=" * 80)

if __name__ == "__main__":
    main()