#!/usr/bin/env python
"""Test GPU Hessian values directly."""

import numpy as np
from pycalphad import Database, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import cupy as cp

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phase_name = 'BCC_A2'

conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

# Create workspace
wks = Workspace(db, components, [phase_name], conditions, verbose=False)

# Generate GPU code
result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
all_device_functions, init_calls, unique_models, phase_map = result

# Test DOF values for Ti-poor phase
test_dof = np.array([1.0, 101325.0, 500.0, 0.94120532, 0.05879468])  # N, P, T, Y(NB), Y(TI)

print("Testing GPU Hessian values...")
print("="*80)
print(f"Test DOF: {test_dof}")
print()

# Create a test kernel
test_kernel_code = all_device_functions + """

extern "C" __global__ void test_hessian_kernel(double* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dof[5] = {1.0, 101325.0, 500.0, 0.94120532, 0.05879468};
        double hess[9];  // 3x3 matrix
        
        pycgpu_model_0_formulahess(hess, dof);
        
        // Store in output for CPU comparison
        for (int i = 0; i < 9; i++) {
            output[i] = hess[i];
        }
    }
}
"""

# Compile and run
print("Compiling test kernel...")
mod = cp.RawModule(code=test_kernel_code)
test_kernel = mod.get_function('test_hessian_kernel')

# Allocate output
output = cp.zeros(9, dtype=cp.float64)

# Run kernel
test_kernel((1,), (1,), (output,))
cp.cuda.Stream.null.synchronize()

# Get results
gpu_hess = output.get()

print("\nGPU Hessian values (flat array):")
for i in range(9):
    print(f"  hess[{i}] = {gpu_hess[i]:12.6e}")

print("\nGPU Hessian as 3x3 matrix:")
gpu_hess_mat = gpu_hess.reshape(3, 3)
for i in range(3):
    print(f"  Row {i}: {gpu_hess_mat[i,0]:12.6e} {gpu_hess_mat[i,1]:12.6e} {gpu_hess_mat[i,2]:12.6e}")

print("\nExpected CPU values from previous output:")
print("  H[T,T]       = -5.215656e-02")
print("  H[T,Y_NB]    = -4.155516e+01")
print("  H[T,Y_TI]    = -6.596052e+01")
print("  H[Y_NB,Y_NB] = 4.416943e+03")
print("  H[Y_NB,Y_TI] = 1.304530e+04")
print("  H[Y_TI,Y_TI] = 7.070793e+04")

print("\nComparing key values:")
print(f"  GPU H[0,0] (T,T)       = {gpu_hess_mat[0,0]:12.6e} vs CPU -5.215656e-02")
print(f"  GPU H[1,1] (Y_NB,Y_NB) = {gpu_hess_mat[1,1]:12.6e} vs CPU 4.416943e+03")
print(f"  GPU H[1,2] (Y_NB,Y_TI) = {gpu_hess_mat[1,2]:12.6e} vs CPU 1.304530e+04")
print(f"  GPU H[2,2] (Y_TI,Y_TI) = {gpu_hess_mat[2,2]:12.6e} vs CPU 7.070793e+04")

print("="*80)