"""
Chunked phase compilation that uses the EXACT same function signatures and structures
as the existing working GPU code.
"""

import os
import hashlib
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from .gpu_codegen import (
    _get_c_define,
    _read_gpu_header,
    _generate_c_code_for_phase_models,
    notebook_model_c_func_name_prefix
)


def compile_many_phases_matching_existing(wks_obj, verbose: bool = False) -> cp.RawModule:
    """
    Compile many phases using the EXACT same structure as the existing working code.
    
    This function:
    1. Uses _generate_c_code_for_phase_models to get the exact same device functions
    2. Includes all the same headers and defines
    3. Uses the exact same PhaseRecord structure and init calls
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    # Generate phase functions using the EXISTING working function
    # This function gets models from wks_obj.models
    result = _generate_c_code_for_phase_models(
        wks_obj, include_hess=True, validate=True
    )
    
    # Unpack the result tuple
    model_functions_c_code = result[0]
    g_phase_record_array_init_calls_c_code = result[1]
    unique_models = result[2]  # This is actually a list of models
    
    # Get the actual number
    num_unique_models = len(unique_models) if isinstance(unique_models, list) else unique_models
    
    if verbose:
        print(f"[GPU] Generated code for {num_unique_models} phases...")
    
    # Read all the required headers (same as existing code)
    svd_c_source = _read_gpu_header("svd.c")
    phase_rec_h_source = _read_gpu_header("phase_rec.h")
    comp_set_h_source = _read_gpu_header("comp_set.h")
    lu_solver_h_source = _read_gpu_header("lu_solver.h")
    minimizer_h_source = _read_gpu_header("minimizer.h")
    eqsolver_h_source = _read_gpu_header("eqsolver.h")
    
    # Get all the required defines
    max_components = _get_c_define("MAX_COMPONENTS")
    max_phases = _get_c_define("MAX_PHASES")
    max_statevars = _get_c_define("MAX_STATEVARS")
    max_dof_per_phase = _get_c_define("MAX_DOF_PER_PHASE")
    max_internal_constraints = _get_c_define("MAX_INTERNAL_CONSTRAINTS")
    max_fixed_mole_fraction_conditions = _get_c_define("MAX_FIXED_MOLE_FRACTION_CONDITIONS")
    min_phase_fraction = _get_c_define("MIN_PHASE_FRACTION")
    
    # Build the complete source (matching existing structure exactly)
    full_source = f"""
// GPU Equilibrium Kernel with {num_unique_models} Phases
// Using exact same structure as existing working code

#include <float.h>
#include <math.h>
#include <stdio.h>

// Required preprocessor defines (must come before headers)
#define MAX_COMPONENTS {max_components}
#define MAX_PHASES {max_phases}
#define MAX_STATEVARS {max_statevars}
#define MAX_DOF_PER_PHASE {max_dof_per_phase}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}
#define MAX_FIXED_MOLE_FRACTION_CONDITIONS {max_fixed_mole_fraction_conditions}
#define MIN_PHASE_FRACTION {min_phase_fraction}

// GPU Debug logging helpers (required by minimizer.h)
__device__ void gpu_debug_log(int segment, const char* message, int condition_idx) {{
    #ifdef VERBOSE_DEBUG
    if (condition_idx >= 0) {{
        printf("[GPU] SEGMENT %02d: %s (condition %d)\\n", segment, message, condition_idx);
    }} else {{
        printf("[GPU] SEGMENT %02d: %s\\n", segment, message);
    }}
    #endif
}}

__device__ void gpu_debug_log_value(const char* message, double value) {{
    #ifdef VERBOSE_DEBUG
    printf("[GPU]   %s: %.15e\\n", message, value);
    #endif
}}

__device__ void gpu_debug_log_array(const char* message, const double* arr, int size) {{
    #ifdef VERBOSE_DEBUG
    printf("[GPU]   %s: [", message);
    for (int i = 0; i < size && i < 5; ++i) {{
        printf("%.6f", arr[i]);
        if (i < size - 1) printf(", ");
    }}
    if (size > 5) printf("...");
    printf("]\\n");
    #endif
}}

// Static library includes (order matters!)
{svd_c_source}
{phase_rec_h_source}
{comp_set_h_source}
{lu_solver_h_source}
{minimizer_h_source}
{eqsolver_h_source}

// Dynamically generated device functions for all phases
{model_functions_c_code}

// Global phase records array
__device__ PhaseRecord g_phase_records_array[{num_unique_models if num_unique_models > 0 else 1}];

// Kernel functions
extern "C" {{

// Initialization kernel (same as existing code)
__global__ void initPhaseRecords() {{
    // Initialize all phase records with their function pointers
    {''.join(g_phase_record_array_init_calls_c_code)}
}}

// Test kernel to verify compilation
__global__ void test_kernel(double* output, const double* input, int n) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {{
        output[tid] = input[tid] * 2.0 + 1.0;
    }}
}}

// Main equilibrium kernel (placeholder for now)
__global__ void equilibrium_kernel(
    const void* global_spec_ptr,
    const void* condition_args_ptr,
    void* results_ptr,
    int num_conditions,
    const void* initial_data_ptr,
    const void* grid_data_ptr
) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize phase records once
    if (tid == 0) {{
        // Call init for all phases
        {''.join(g_phase_record_array_init_calls_c_code)}
    }}
    __syncthreads();
    
    // Main equilibrium calculation would go here
    // For now just verify we compiled successfully
    if (tid == 0 && results_ptr != nullptr) {{
        double* results = (double*)results_ptr;
        results[0] = {num_unique_models}.0;  // Return number of phases as verification
    }}
}}

}} // extern "C"
"""
    
    if verbose:
        print(f"[GPU] Source code size: {len(full_source)} characters")
        print(f"[GPU] Compiling with nvcc backend...")
    
    # Compile with nvcc (which supports all C++ features)
    try:
        start_time = time.time()
        
        module = cp.RawModule(
            code=full_source,
            options=('-std=c++11', '-O3'),
            backend='nvcc'
        )
        
        compile_time = time.time() - start_time
        
        if verbose:
            print(f"[GPU] Compilation successful! Time: {compile_time:.2f} seconds")
            print(f"[GPU] {num_unique_models} phases compiled and available")
        
        return module
        
    except Exception as e:
        print(f"[GPU] Compilation failed: {e}")
        
        # Save source for debugging
        debug_file = "gpu_debug_source.cu"
        with open(debug_file, 'w') as f:
            f.write(full_source)
        print(f"[GPU] Source saved to {debug_file} for debugging")
        
        raise


def test_with_real_system():
    """Test compilation with a real thermodynamic system."""
    
    if not GPU_AVAILABLE:
        print("GPU not available")
        return
    
    print("=" * 80)
    print("TESTING MULTI-PHASE COMPILATION WITH REAL SYSTEM")
    print("=" * 80)
    
    from pycalphad import Database
    from pycalphad.core.workspace import Workspace
    import pycalphad.variables as v
    
    # Use the Al-Cu-Fe database which has many phases
    try:
        dbf = Database('Al-Cu-Fe.tdb')
    except FileNotFoundError:
        print("Al-Cu-Fe.tdb not found, using test database")
        # Create a simple test database
        dbf = Database()
        dbf.elements = ['AL', 'CU', 'FE', 'VA']
        # Add multiple test phases
        for i in range(10):  # Create 10 test phases
            phase_name = f'PHASE_{i}'
            dbf.add_phase(phase_name, 'GEM', [['AL', 'CU', 'FE', 'VA']])
            dbf.add_phase_constituents(phase_name, [['AL', 'CU', 'FE', 'VA']])
            # Add simple parameters
            dbf.add_parameter('GES', phase_name, [['AL'], ['VA']], 0, -1000.0 * (i+1))
            dbf.add_parameter('GES', phase_name, [['CU'], ['VA']], 0, -2000.0 * (i+1))
            dbf.add_parameter('GES', phase_name, [['FE'], ['VA']], 0, -3000.0 * (i+1))
    
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = list(dbf.phases.keys())
    
    print(f"Database has {len(phases)} phases: {phases}")
    
    # Create workspace with conditions
    conditions = {
        v.T: 800,
        v.P: 101325,
        v.X('AL'): 0.3,
        v.X('CU'): 0.3
    }
    
    # Create workspace
    wks = Workspace(dbf, comps, phases, conditions, verbose=True)
    
    # Test compilation
    try:
        module = compile_many_phases_matching_existing(wks, verbose=True)
        
        print("\n" + "=" * 80)
        print("SUCCESS: Compilation completed!")
        print("=" * 80)
        
        # Test the kernel
        test_kernel = module.get_function("test_kernel")
        
        n = 10
        input_data = cp.arange(n, dtype=cp.float64)
        output_data = cp.zeros(n, dtype=cp.float64)
        
        test_kernel((1,), (n,), (output_data, input_data, n))
        
        result = output_data.get()
        expected = input_data.get() * 2.0 + 1.0
        
        if np.allclose(result, expected):
            print("✓ Test kernel works correctly!")
        else:
            print("✗ Test kernel failed!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_with_real_system()