#!/usr/bin/env python
"""Test with the correct CPU matrix that achieves X(TI)=0.9."""

import numpy as np
import cupy as cp

print("CORRECT CPU MATRIX TEST")
print("=" * 60)

# Current state: phase_amount=1.0, phase_X_TI=0.903147
# Target: overall_X_TI = 0.9
# Required: phase_amount * phase_X_TI = 0.9

current_phase_amount = 1.0
current_X_TI = 0.903147
target_X_TI = 0.9

# Calculate the correct phase amount change
required_phase_amount = target_X_TI / current_X_TI
phase_amount_change = required_phase_amount - current_phase_amount

print(f"Current: NP={current_phase_amount:.6f}, X(TI)={current_X_TI:.6f}")
print(f"Target overall X(TI): {target_X_TI:.6f}")
print(f"Required NP: {required_phase_amount:.10f}")
print(f"Required change: {phase_amount_change:.10f}")

# The constraint equation: NP * X_TI = target
# Current violation: 1.0 * 0.903147 - 0.9 = 0.003147
mass_residual = current_phase_amount * current_X_TI - target_X_TI

print(f"Mass residual: {mass_residual:.6f}")

# The CPU matrix equation is:
# For the mass balance constraint row: [0, 0, X_TI] * [dmu_NB, dmu_TI, dNP] = target_change
# where target_change accounts for the residual

# The RHS should be constructed so that the solution gives the right phase change
# If we want dNP = -0.003485 (the required change)
# And the matrix row is [0, 0, 0.903147]
# Then RHS = 0.903147 * (-0.003485) = -0.003147

target_rhs_for_mass = current_X_TI * phase_amount_change

print(f"\nCorrect matrix construction:")
print(f"Matrix row: [0, 0, {current_X_TI:.6f}]")
print(f"Required solution[2]: {phase_amount_change:.10f}")
print(f"Therefore RHS[2]: {target_rhs_for_mass:.10f}")

# Test this matrix
constraint_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], 
    [0.0, 0.0, current_X_TI]
])

rhs = np.array([
    0.0,
    0.0, 
    target_rhs_for_mass
])

print(f"\nMatrix:\n{constraint_matrix}")
print(f"RHS: {rhs}")

solution = np.linalg.solve(constraint_matrix, rhs)
print(f"Solution: {solution}")

# Verify the result
new_phase_amount = current_phase_amount + solution[2]
final_X_TI = new_phase_amount * current_X_TI

print(f"\nVerification:")
print(f"New phase amount: {new_phase_amount:.10f}")
print(f"Final X(TI): {final_X_TI:.10f}")
print(f"Target: {target_X_TI:.10f}")
print(f"Error: {abs(final_X_TI - target_X_TI):.2e}")

# Now let's see how the CPU actually constructs this RHS
print(f"\n" + "="*60)
print("HOW CPU CONSTRUCTS THE RHS")

# From minimizer code: equilibrium_rhs[row] -= mass_residual
# So the RHS starts as some target, then the residual is subtracted

# If the target is 0.9 (the desired overall composition)
# and we subtract the residual (0.003147)
# we get: 0.9 - 0.003147 = 0.896853

cpu_style_rhs = target_X_TI - mass_residual
print(f"CPU style RHS: {target_X_TI:.6f} - {mass_residual:.6f} = {cpu_style_rhs:.6f}")

# Test with CPU-style RHS  
rhs_cpu_style = np.array([0.0, 0.0, cpu_style_rhs])
solution_cpu_style = np.linalg.solve(constraint_matrix, rhs_cpu_style)

print(f"CPU-style solution: {solution_cpu_style}")

new_phase_amount_cpu = current_phase_amount + solution_cpu_style[2] 
final_X_TI_cpu = new_phase_amount_cpu * current_X_TI

print(f"CPU-style result:")
print(f"  New phase amount: {new_phase_amount_cpu:.10f}")
print(f"  Final X(TI): {final_X_TI_cpu:.10f}")

# This still gives wrong answer. The issue might be in the matrix structure.
# Let me try a different approach - what if the matrix row is different?

print(f"\n" + "="*60)
print("ALTERNATIVE MATRIX STRUCTURE")

# What if the constraint is formulated as: 
# change_in_overall_composition = target - current
# And the matrix multiplies [dmu_NB, dmu_TI, dNP] to give this change

desired_change = target_X_TI - (current_phase_amount * current_X_TI)
print(f"Desired change in overall X(TI): {desired_change:.10f}")

# If dNP changes the overall composition by: dNP * current_X_TI
# Then we want: dNP * current_X_TI = desired_change
# So: dNP = desired_change / current_X_TI

required_dNP = desired_change / current_X_TI
print(f"Required dNP: {required_dNP:.10f}")

# Matrix equation: [0, 0, current_X_TI] * [0, 0, dNP] = desired_change
alternative_rhs = np.array([0.0, 0.0, desired_change])
solution_alt = np.linalg.solve(constraint_matrix, alternative_rhs)

print(f"Alternative solution: {solution_alt}")

new_phase_amount_alt = current_phase_amount + solution_alt[2]
final_X_TI_alt = new_phase_amount_alt * current_X_TI

print(f"Alternative result:")
print(f"  New phase amount: {new_phase_amount_alt:.10f}")  
print(f"  Final X(TI): {final_X_TI_alt:.10f}")
print(f"  Error: {abs(final_X_TI_alt - target_X_TI):.2e}")

if abs(final_X_TI_alt - target_X_TI) < 1e-10:
    print("✓ This approach works!")
    
    # Test GPU solver with this correct matrix
    print(f"\nTesting GPU solver...")
    
    gpu_code = '''
    extern "C" __global__ void solve_3x3(double* A, double* b, double* x) {
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
        kernel = module.get_function('solve_3x3')
        
        d_A = cp.asarray(constraint_matrix.flatten(), dtype=cp.float64)
        d_b = cp.asarray(alternative_rhs, dtype=cp.float64)
        d_x = cp.zeros(3, dtype=cp.float64)
        
        kernel((1,), (1,), (d_A, d_b, d_x))
        gpu_solution = d_x.get()
        
        print(f"CPU solution: {solution_alt}")
        print(f"GPU solution: {gpu_solution}")
        
        if np.allclose(solution_alt, gpu_solution, atol=1e-12):
            print("✓ GPU matches CPU exactly!")
        else:
            print("✗ GPU differs from CPU")
            print(f"Difference: {np.abs(solution_alt - gpu_solution)}")
            
    except Exception as e:
        print(f"GPU test failed: {e}")
else:
    print("✗ Still not the right approach")