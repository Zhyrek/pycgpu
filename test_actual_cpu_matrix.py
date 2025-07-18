#!/usr/bin/env python
"""Test with the actual matrix structure CPU uses to reach X(TI)=0.9."""

import numpy as np
import cupy as cp

print("TESTING ACTUAL CPU MATRIX STRUCTURE")
print("=" * 60)

# From CPU trace: it goes from X(TI)=0.903147 to exactly X(TI)=0.900000
# The key insight is that the CPU must be solving a constraint where:
# phase_amount * phase_composition = target_composition

current_X_TI = 0.903147
target_X_TI = 0.9
current_phase_amount = 1.0

# The CPU constraint equation:
# For single phase: NP * Y_TI = X_TI_target 
# Where Y_TI is the site fraction in the phase
# and X_TI is the overall mole fraction

print("Current state:")
print(f"  Phase amount: {current_phase_amount}")
print(f"  Phase X(TI): {current_X_TI}")
print(f"  Overall X(TI): {current_phase_amount * current_X_TI}")
print(f"  Target X(TI): {target_X_TI}")

# The residual is the constraint violation
mass_residual = current_phase_amount * current_X_TI - target_X_TI
print(f"  Mass residual: {mass_residual:.6f}")

# From minimizer code, the constraint row is:
# [0, 0, ..., phase_composition] for the mass balance
# This multiplies [mu_changes..., phase_amount_change] = target_composition

# But the key is what the CPU does with phase amounts vs compositions
# Let's examine if CPU changes the phase composition instead of amount

print("\nTesting CPU's constraint handling...")

# Option 1: CPU changes phase amount to satisfy constraint
required_phase_amount = target_X_TI / current_X_TI
phase_amount_change = required_phase_amount - current_phase_amount

print(f"\nOption 1 - Change phase amount:")
print(f"  Required phase amount: {required_phase_amount:.10f}")
print(f"  Phase amount change: {phase_amount_change:.10f}")
print(f"  New overall X(TI): {required_phase_amount * current_X_TI:.10f}")

# Option 2: CPU changes phase composition to satisfy constraint  
required_phase_X_TI = target_X_TI / current_phase_amount
phase_X_TI_change = required_phase_X_TI - current_X_TI

print(f"\nOption 2 - Change phase composition:")
print(f"  Required phase X(TI): {required_phase_X_TI:.10f}")
print(f"  Phase X(TI) change: {phase_X_TI_change:.10f}")
print(f"  New overall X(TI): {current_phase_amount * required_phase_X_TI:.10f}")

# The CPU trace showed exactly X(TI)=0.9, so let's see which approach works
print(f"\nWhich approach gives exactly 0.900000?")
print(f"Option 1 result: {required_phase_amount * current_X_TI:.10f}")
print(f"Option 2 result: {current_phase_amount * required_phase_X_TI:.10f}")

# Both should give 0.9, but the question is which variable the CPU actually changes

# Let's construct the matrix the CPU actually uses
# From the minimizer code, we know it has chemical potential variables
# plus phase amount variables

print("\n" + "="*60)
print("CONSTRUCTING ACTUAL CPU EQUILIBRIUM MATRIX")

# After consolidation, we have:
# - 2 free chemical potentials (NB, TI) 
# - 1 phase amount variable
# - 1 mass balance constraint

# The matrix structure is approximately:
# | ∂G/∂μ_NB ∂G/∂μ_TI  ∂G/∂NP |   |Δμ_NB|   |0|
# |    0        0        X_TI  | * |Δμ_TI| = |target-current|
# |   ...     ...       ...   |   |ΔNP  |   |...|

# For the mass balance constraint row:
constraint_matrix = np.array([
    [1.0, 0.0, 0.0],           # Some chemical potential constraint
    [0.0, 1.0, 0.0],           # Another chemical potential constraint  
    [0.0, 0.0, current_X_TI]   # Mass balance: NP * X_TI = target
])

# The RHS includes the residual correction
rhs = np.array([
    0.0,  # Chemical potential updates (small near convergence)
    0.0,  # Chemical potential updates
    target_X_TI - mass_residual  # Target minus current residual
])

print("Equilibrium matrix:")
print(constraint_matrix)
print(f"RHS: {rhs}")

# Solve the system
solution = np.linalg.solve(constraint_matrix, rhs)
print(f"Solution: {solution}")

# Apply the update
new_phase_amount = current_phase_amount + solution[2] 
final_X_TI = new_phase_amount * current_X_TI

print(f"\nAfter update:")
print(f"  New phase amount: {new_phase_amount:.10f}")
print(f"  Final X(TI): {final_X_TI:.10f}")
print(f"  Target: {target_X_TI:.10f}")
print(f"  Error: {abs(final_X_TI - target_X_TI):.2e}")

# Test the GPU solver with this exact matrix
print("\n" + "="*60) 
print("TESTING GPU SOLVER WITH THIS MATRIX")

gpu_code = '''
extern "C" __global__ void test_matrix_solve(double* A, double* b, double* x) {
    // Solve 3x3 system using direct inversion
    double det = A[0]*(A[4]*A[8] - A[5]*A[7]) - A[1]*(A[3]*A[8] - A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
    
    if (fabs(det) < 1e-16) {
        x[0] = x[1] = x[2] = 0.0;
        return;
    }
    
    double inv_det = 1.0 / det;
    x[0] = inv_det * (b[0]*(A[4]*A[8] - A[5]*A[7]) - b[1]*(A[1]*A[8] - A[2]*A[7]) + b[2]*(A[1]*A[5] - A[2]*A[4]));
    x[1] = inv_det * (b[1]*(A[0]*A[8] - A[2]*A[6]) - b[0]*(A[3]*A[8] - A[5]*A[6]) + b[2]*(A[3]*A[2] - A[0]*A[5]));
    x[2] = inv_det * (b[2]*(A[0]*A[4] - A[1]*A[3]) - b[0]*(A[6]*A[4] - A[7]*A[3]) + b[1]*(A[6]*A[1] - A[7]*A[0]));
}
'''

try:
    module = cp.RawModule(code=gpu_code)
    kernel = module.get_function('test_matrix_solve')
    
    d_A = cp.asarray(constraint_matrix.flatten(), dtype=cp.float64)
    d_b = cp.asarray(rhs, dtype=cp.float64)  
    d_x = cp.zeros(3, dtype=cp.float64)
    
    kernel((1,), (1,), (d_A, d_b, d_x))
    gpu_solution = d_x.get()
    
    print(f"CPU solution: {solution}")
    print(f"GPU solution: {gpu_solution}")
    print(f"Difference: {np.abs(solution - gpu_solution)}")
    
    if np.allclose(solution, gpu_solution, atol=1e-12):
        print("✓ GPU solver matches CPU!")
    else:
        print("✗ GPU solver differs from CPU!")
        
except Exception as e:
    print(f"GPU test failed: {e}")