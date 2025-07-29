#!/usr/bin/env python
"""Simpler test of gradient values."""

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

# Test DOF values
test_dof = np.array([1.0, 101325.0, 500.0, 0.93877551, 0.06122449])  # N, P, T, Y(NB), Y(TI)

print("Testing gradient function output order...")
print("="*80)
print(f"Test DOF: {test_dof}")
print()

# Create a simple test kernel without cstdio
test_kernel_code = all_device_functions + """

extern "C" __global__ void test_gradient_kernel(double* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dof[5] = {1.0, 101325.0, 500.0, 0.93877551, 0.06122449};
        double grad[3];
        
        pycgpu_model_0_formulagrad(grad, dof);
        
        // Store in output for CPU comparison
        output[0] = grad[0];
        output[1] = grad[1];
        output[2] = grad[2];
    }
}
"""

# Compile and run
print("Compiling test kernel...")
mod = cp.RawModule(code=test_kernel_code)
test_kernel = mod.get_function('test_gradient_kernel')

# Allocate output
output = cp.zeros(3, dtype=cp.float64)

# Run kernel
test_kernel((1,), (1,), (output,))
cp.cuda.Stream.null.synchronize()

# Get results
gpu_grad = output.get()

print("\nGPU gradient values:")
print(f"  Temperature derivative: {gpu_grad[0]}")
print(f"  Y(NB) derivative: {gpu_grad[1]}")
print(f"  Y(TI) derivative: {gpu_grad[2]}")

# Compare with expected CPU values from debug output
print("\nExpected CPU gradient values (from debug output):")
print(f"  Temperature derivative: -51.3634249")
print(f"  Y(NB) derivative: -14863.1786191")
print(f"  Y(TI) derivative: -8784.46966485")

print("\nDifferences:")
print(f"  T diff: {abs(gpu_grad[0] - (-51.3634249))}")
print(f"  Y(NB) diff: {abs(gpu_grad[1] - (-14863.1786191))}")
print(f"  Y(TI) diff: {abs(gpu_grad[2] - (-8784.46966485))}")

print("="*80)