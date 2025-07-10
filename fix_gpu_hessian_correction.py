#!/usr/bin/env python3
"""Propose a correction for GPU hessian values"""

import numpy as np

# The issue: GPU hessian diagonal elements include spurious cross-terms
# GPU: d²G/dY_i² = RT/Y_i + RT/Y_j (where j ≠ i)
# CPU: d²G/dY_i² = RT/Y_i (correct)

# For a binary system with components i and j:
# The correction is to subtract RT/Y_j from the diagonal element [i,i]

def correct_gpu_hessian(hessian, dof, num_statevars, phase_dof):
    """
    Correct GPU hessian by removing spurious entropy cross-terms.
    
    Parameters:
    -----------
    hessian : ndarray
        The GPU-computed hessian matrix
    dof : ndarray
        Degrees of freedom [N, P, T, Y1, Y2, ...]
    num_statevars : int
        Number of state variables (typically 3 for N, P, T)
    phase_dof : int
        Number of phase degrees of freedom (site fractions)
    
    Returns:
    --------
    corrected_hessian : ndarray
        Corrected hessian matrix
    """
    R = 8.3145
    T = dof[2]  # Temperature
    
    # Make a copy to avoid modifying original
    corrected = hessian.copy()
    
    # For each site fraction variable
    for i in range(phase_dof):
        Yi_idx = num_statevars + i
        Yi = dof[Yi_idx]
        
        # Calculate the spurious contribution from other site fractions
        spurious_sum = 0.0
        for j in range(phase_dof):
            if i != j:
                Yj_idx = num_statevars + j
                Yj = dof[Yj_idx]
                if Yj > 1e-15:  # Avoid division by zero
                    spurious_sum += R * T / Yj
        
        # Subtract the spurious contribution from diagonal element
        corrected[Yi_idx, Yi_idx] -= spurious_sum
    
    return corrected

# Test the correction
print("=== Testing Hessian Correction ===")

# Test values
dof = np.array([1.0, 101325.0, 1000.0, 0.612245, 0.387755])
num_statevars = 3
phase_dof = 2

# GPU values (from our test)
gpu_hess_33 = 35023.011274  # RT/Y_NB + RT/Y_TI
gpu_hess_44 = 56465.675  # Would be RT/Y_TI + RT/Y_NB

# Expected CPU values
cpu_hess_33 = 13580.347737  # RT/Y_NB only
cpu_hess_44 = 21442.663538  # RT/Y_TI only

# Create mock hessian
gpu_hessian = np.zeros((5, 5))
gpu_hessian[3, 3] = gpu_hess_33
gpu_hessian[4, 4] = gpu_hess_44  # Assume same issue

print(f"GPU hessian[3,3] = {gpu_hess_33:.2f}")
print(f"Expected CPU[3,3] = {cpu_hess_33:.2f}")

# Apply correction
corrected_hessian = correct_gpu_hessian(gpu_hessian, dof, num_statevars, phase_dof)

print(f"\nCorrected hessian[3,3] = {corrected_hessian[3,3]:.2f}")
print(f"Error after correction = {abs(corrected_hessian[3,3] - cpu_hess_33):.6f}")

# The correction formula for binary system:
# corrected[i,i] = gpu[i,i] - RT/Y_j
# where j is the other component

print("\n=== Simple Correction Formula ===")
print("For binary system with Y_NB and Y_TI:")
print("corrected_hess[3,3] = gpu_hess[3,3] - RT/Y_TI")
print("corrected_hess[4,4] = gpu_hess[4,4] - RT/Y_NB")