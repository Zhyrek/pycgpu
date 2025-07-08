#!/usr/bin/env python3
"""Test just the GPU energy calculation to see if it's using the correct function"""

import numpy as np
import cupy as cp
from pycalphad import Database, Model, variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models, _generate_full_gpu_source
from pycalphad.gpu.gpu_codegen import compute_dynamic_kernel_sizes

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing GPU Energy Calculation ===")

# Create workspace
wks = Workspace(db, comps, phases, eq_conditions)

# Generate GPU code
model_funcs_c, pr_init_calls_c, unique_py_models, py_phase_name_to_unique_idx_map = \
    _generate_c_code_for_phase_models(wks, include_hess=False, validate=False)

# Generate full GPU source
dynamic_sizes = compute_dynamic_kernel_sizes(wks)
full_gpu_source = _generate_full_gpu_source(wks, model_funcs_c, pr_init_calls_c, len(unique_py_models))

# Create a simple test kernel
test_kernel_source = f"""
{full_gpu_source}

extern "C" {{
    __global__ void test_energy_kernel(double* results, double* dof) {{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid == 0) {{
            // Test with the exact DOF values from the test
            double test_dof[5] = {{1.0, 101325.0, 1000.0, 0.612244897959181, 0.387755102040819}};
            results[0] = pycgpu_model_0_obj(test_dof);
            
            // Also test with the second phase
            double test_dof2[5] = {{1.0, 101325.0, 1000.0, 0.593664271503332, 0.406335728496668}};
            results[1] = pycgpu_model_0_obj(test_dof2);
        }}
    }}
}}
"""

print("=== Compiling test kernel ===")
try:
    test_module = cp.RawModule(code=test_kernel_source, options=('-std=c++14',))
    test_kernel = test_module.get_function('test_energy_kernel')
    
    # Allocate memory for results
    results = cp.zeros(2, dtype=cp.float64)
    dof = cp.zeros(5, dtype=cp.float64)
    
    # Launch kernel
    test_kernel((1,), (1,), (results, dof))
    
    # Get results back to CPU
    cpu_results = results.get()
    
    print(f"GPU obj function result for phase 0: {cpu_results[0]:.6f} J/mol")
    print(f"GPU obj function result for phase 1: {cpu_results[1]:.6f} J/mol")
    
    # Compare with expected values
    expected_0 = -49817.155997
    expected_1 = -49734.966498
    
    print(f"Expected phase 0: {expected_0:.6f} J/mol")
    print(f"Expected phase 1: {expected_1:.6f} J/mol")
    
    diff_0 = abs(cpu_results[0] - expected_0)
    diff_1 = abs(cpu_results[1] - expected_1)
    
    print(f"Difference phase 0: {diff_0:.6f} J/mol")
    print(f"Difference phase 1: {diff_1:.6f} J/mol")
    
    if diff_0 < 0.001 and diff_1 < 0.001:
        print("✓ SUCCESS: GPU energy functions are correct!")
    else:
        print("✗ FAIL: GPU energy functions are still incorrect")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()