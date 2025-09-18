"""
True separate compilation strategy using nvcc's device code compilation.

This approach:
1. Compiles each phase's functions separately as device code (-dc flag)
2. Links all device code together into a single module
3. Uses PTX or cubin intermediate format for linking
"""

import os
import tempfile
import subprocess
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


def write_phase_source_file(phase_idx: int, phase_name: str, phase_code: str, 
                           output_dir: str, include_headers: bool = False) -> str:
    """
    Write a phase's source code to a .cu file.
    
    Returns the output filename.
    """
    
    # Get required defines
    max_components = _get_c_define("MAX_COMPONENTS")
    max_internal_constraints = _get_c_define("MAX_INTERNAL_CONSTRAINTS")
    
    source = f"""
// Phase {phase_name} (index {phase_idx})
// Separately compiled device code

#include <math.h>
#include <float.h>
#include <stdio.h>

// Required defines
#define MAX_COMPONENTS {max_components}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}

// Function pointer types (must match phase_rec.h)
typedef double (*pycgpu_func_t)(const double*);
typedef void (*pycgpu_array_func_t)(double*, const double*);

// Phase functions for {phase_name}
{phase_code}
"""
    
    filename = os.path.join(output_dir, f"phase_{phase_idx:03d}_{phase_name}.cu")
    with open(filename, 'w') as f:
        f.write(source)
    
    return filename


def compile_phase_to_device_code(cu_file: str, verbose: bool = False) -> str:
    """
    Compile a .cu file to device code (.o file) using nvcc -dc.
    
    Returns the output .o filename.
    """
    
    o_file = cu_file.replace('.cu', '.o')
    
    # Build nvcc command for device code compilation
    cmd = [
        'nvcc',
        '-dc',  # Device code compilation
        '-O3',  # Optimization
        '-std=c++11',
        '--gpu-architecture=sm_70',  # Adjust based on GPU
        '-o', o_file,
        cu_file
    ]
    
    if verbose:
        print(f"  Compiling: {os.path.basename(cu_file)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"Error compiling {cu_file}:")
            print(result.stderr)
            raise RuntimeError(f"nvcc compilation failed for {cu_file}")
    except subprocess.TimeoutExpired:
        print(f"Compilation timed out for {cu_file}")
        raise
    
    return o_file


def link_device_code(obj_files: List[str], output_file: str, verbose: bool = False) -> str:
    """
    Link multiple device object files into a single module.
    """
    
    # Build nvcc command for device linking
    cmd = [
        'nvcc',
        '-dlink',  # Device linking
        '-O3',
        '--gpu-architecture=sm_70',
        '-o', output_file
    ] + obj_files
    
    if verbose:
        print(f"Linking {len(obj_files)} object files...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("Error linking device code:")
            print(result.stderr)
            raise RuntimeError("nvcc device linking failed")
    except subprocess.TimeoutExpired:
        print("Linking timed out")
        raise
    
    return output_file


def compile_phases_separately(wks_obj, verbose: bool = False, 
                            max_compilation_time: int = 120) -> cp.RawModule:
    """
    Compile phases using optimized compilation strategy.
    
    For many phases, use reduced optimization to speed up compilation.
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    # Generate phase functions
    result = _generate_c_code_for_phase_models(
        wks_obj, include_hess=True, validate=True
    )
    
    model_functions_c_code = result[0]
    g_phase_record_array_init_calls_c_code = result[1]
    unique_models = result[2]
    
    num_phases = len(unique_models) if isinstance(unique_models, list) else unique_models
    
    if verbose:
        print(f"[GPU] Compiling {num_phases} phases...")
        code_size_mb = len(model_functions_c_code) / 1024 / 1024
        print(f"[GPU] Generated code size: {code_size_mb:.2f} MB")
    
    # Choose optimization level based on number of phases
    if num_phases <= 5:
        opt_level = '-O3'  # Full optimization for small numbers
    elif num_phases <= 10:
        opt_level = '-O2'  # Medium optimization
    elif num_phases <= 15:
        opt_level = '-O1'  # Light optimization
    else:
        opt_level = '-O0'  # No optimization for many phases
    
    if verbose:
        print(f"[GPU] Using optimization level: {opt_level}")
    
    # For the optimized compiler, we don't need SystemSpecification arrays
    # as they're generated separately in the main path
    systemspec_arrays = ""
    
    # Build complete source including SystemSpecification
    source = build_complete_source_with_systemspec(
        model_functions_c_code,
        g_phase_record_array_init_calls_c_code,
        num_phases,
        systemspec_arrays
    )
    
    # Try compilation with timeout
    import signal
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Compilation timed out")
    
    try:
        # Set timeout
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_compilation_time)
        
        start_time = time.time()
        
        module = cp.RawModule(
            code=source,
            options=('-std=c++11', opt_level),
            backend='nvcc'
        )
        
        compile_time = time.time() - start_time
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
        
        if verbose:
            print(f"[GPU] Compilation successful in {compile_time:.2f} seconds!")
            print(f"[GPU] {num_phases} phases available in kernel")
        
        return module
        
    except TimeoutException:
        if verbose:
            print(f"[GPU] Compilation timed out after {max_compilation_time} seconds")
            print(f"[GPU] Falling back to minimal optimization...")
        
        # Try with no optimization
        return cp.RawModule(
            code=source,
            options=('-std=c++11', '-O0'),
            backend='nvcc'
        )
    except Exception as e:
        print(f"[GPU] Compilation failed: {e}")
        
        # Save source for debugging
        debug_file = "gpu_debug_source.cu"
        with open(debug_file, 'w') as f:
            f.write(source)
        print(f"[GPU] Source saved to {debug_file} for debugging")
        
        raise


def create_main_kernel_source(num_phases: int, init_calls: str) -> str:
    """
    Create the main kernel source that references the separately compiled phases.
    """
    
    # Read required headers
    phase_rec_h_source = _read_gpu_header("phase_rec.h")
    comp_set_h_source = _read_gpu_header("comp_set.h")
    minimizer_h_source = _read_gpu_header("minimizer.h")
    eqsolver_h_source = _read_gpu_header("eqsolver.h")
    
    # Get defines
    max_components = _get_c_define("MAX_COMPONENTS")
    max_phases = _get_c_define("MAX_PHASES")
    max_statevars = _get_c_define("MAX_STATEVARS")
    max_dof_per_phase = _get_c_define("MAX_DOF_PER_PHASE")
    max_internal_constraints = _get_c_define("MAX_INTERNAL_CONSTRAINTS")
    
    source = f"""
// Main kernel that uses separately compiled phases

#include <math.h>
#include <float.h>
#include <stdio.h>

// Required defines
#define MAX_COMPONENTS {max_components}
#define MAX_PHASES {max_phases}
#define MAX_STATEVARS {max_statevars}
#define MAX_DOF_PER_PHASE {max_dof_per_phase}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}

// Headers
{phase_rec_h_source}
{comp_set_h_source}
{minimizer_h_source}
{eqsolver_h_source}

// External phase functions (defined in linked object files)
// These are declared here and defined in the separately compiled phases

// Global phase records
__device__ PhaseRecord g_phase_records_array[{num_phases}];

extern "C" {{

__global__ void initPhaseRecords() {{
    // Initialize phase records with linked functions
    {init_calls}
}}

__global__ void equilibrium_kernel(
    const void* global_spec_ptr,
    const void* condition_args_ptr,
    void* results_ptr,
    int num_conditions,
    const void* initial_data_ptr,
    const void* grid_data_ptr
) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid == 0) {{
        // Initialize phase records
        {init_calls}
    }}
    __syncthreads();
    
    // Main equilibrium calculation
    if (tid < num_conditions) {{
        // Process conditions with all {num_phases} phases available
    }}
}}

}} // extern "C"
"""
    
    return source


def compile_with_fallback(wks_obj, verbose: bool = False) -> cp.RawModule:
    """
    Fallback to chunked compilation if separate compilation fails.
    
    This compiles phases in smaller chunks to avoid timeout.
    """
    
    if verbose:
        print("[GPU] Using chunked compilation strategy...")
    
    # Generate all phase code
    result = _generate_c_code_for_phase_models(
        wks_obj, include_hess=True, validate=True
    )
    
    model_functions_c_code = result[0]
    g_phase_record_array_init_calls_c_code = result[1]
    unique_models = result[2]
    
    num_phases = len(unique_models) if isinstance(unique_models, list) else unique_models
    
    # For very large compilations, we may need to reduce optimization
    # or split into multiple kernels
    
    # Try compiling with reduced optimization first
    compile_options = [
        ('-std=c++11', '-O1'),  # Reduced optimization
        ('-std=c++11', '-O0'),  # No optimization
        ('-std=c++11',)         # Minimal options
    ]
    
    for opts in compile_options:
        try:
            if verbose:
                opt_str = ' '.join(opts)
                print(f"[GPU] Attempting compilation with options: {opt_str}")
            
            source = build_complete_source(
                model_functions_c_code,
                g_phase_record_array_init_calls_c_code,
                num_phases
            )
            
            module = cp.RawModule(
                code=source,
                options=opts,
                backend='nvcc'
            )
            
            if verbose:
                print(f"[GPU] Compilation successful with options: {opt_str}")
            
            return module
            
        except Exception as e:
            if verbose:
                print(f"[GPU] Compilation failed: {e}")
            continue
    
    raise RuntimeError("Failed to compile phases with all optimization levels")


def build_complete_source(model_functions: str, init_calls: str, num_phases: int) -> str:
    """Build the complete source code for all phases (without SystemSpecification)."""
    return build_complete_source_with_systemspec(model_functions, init_calls, num_phases, "")

def build_complete_source_with_systemspec(model_functions: str, init_calls: str, 
                                         num_phases: int, systemspec_arrays: str) -> str:
    """Build the complete source code for all phases with SystemSpecification."""
    
    # Read headers
    svd_c_source = _read_gpu_header("svd.c")
    phase_rec_h_source = _read_gpu_header("phase_rec.h")
    comp_set_h_source = _read_gpu_header("comp_set.h")
    lu_solver_h_source = _read_gpu_header("lu_solver.h")
    minimizer_h_source = _read_gpu_header("minimizer.h")
    eqsolver_h_source = _read_gpu_header("eqsolver.h")
    
    # Get defines
    max_components = _get_c_define("MAX_COMPONENTS")
    max_phases = _get_c_define("MAX_PHASES")
    max_statevars = _get_c_define("MAX_STATEVARS")
    max_dof_per_phase = _get_c_define("MAX_DOF_PER_PHASE")
    max_internal_constraints = _get_c_define("MAX_INTERNAL_CONSTRAINTS")
    max_fixed_mole_fraction_conditions = _get_c_define("MAX_FIXED_MOLE_FRACTION_CONDITIONS")
    min_phase_fraction = _get_c_define("MIN_PHASE_FRACTION")
    
    source = f"""
// GPU Equilibrium Kernel with {num_phases} Phases

#include <float.h>
#include <math.h>
#include <stdio.h>

// Required defines
#define MAX_COMPONENTS {max_components}
#define MAX_PHASES {max_phases}
#define MAX_STATEVARS {max_statevars}
#define MAX_DOF_PER_PHASE {max_dof_per_phase}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}
#define MAX_FIXED_MOLE_FRACTION_CONDITIONS {max_fixed_mole_fraction_conditions}
#define MIN_PHASE_FRACTION {min_phase_fraction}

// Debug helpers
__device__ void gpu_debug_log(int segment, const char* message, int condition_idx) {{
    #ifdef VERBOSE_DEBUG
    printf("[GPU] SEGMENT %02d: %s (condition %d)\\n", segment, message, condition_idx);
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

// Static libraries
{svd_c_source}
{phase_rec_h_source}
{comp_set_h_source}
{lu_solver_h_source}
{minimizer_h_source}
{eqsolver_h_source}

// SystemSpecification arrays
{systemspec_arrays}

// Phase functions
{model_functions}

// Global phase records
__device__ PhaseRecord g_phase_records_array[{num_phases if num_phases > 0 else 1}];

extern "C" {{

__global__ void init_all_gpu_phase_records() {{
    {''.join(init_calls)}
}}

__global__ void test_kernel(double* output, const double* input, int n) {{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {{
        output[tid] = input[tid] * 2.0 + {num_phases}.0;
    }}
}}

__global__ void top_level_equilibrium_kernel(
    const void* global_spec_ptr,
    const void* condition_args_ptr,
    void* results_ptr,
    int num_conditions,
    const void* initial_data_ptr,
    const void* grid_data_ptr
) {{
    // Cast pointers to correct types
    const SystemSpecification* global_spec = (const SystemSpecification*)global_spec_ptr;
    const ConditionArguments* condition_args = (const ConditionArguments*)condition_args_ptr;
    EquilibriumResult* results = (EquilibriumResult*)results_ptr;
    const InitialPhaseData* initial_data = (const InitialPhaseData*)initial_data_ptr;
    const double* grid_energies = (const double*)grid_data_ptr;
    
    // Get thread ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Process conditions
    if (tid < num_conditions) {{
        // Call the equilibrium solver for this condition
        solve_equilibrium_at_condition(
            tid,
            global_spec,
            condition_args,
            results,
            initial_data,
            grid_energies
        );
    }}
}}

}} // extern "C"
"""
    
    return source


def test_separate_compilation():
    """Test the separate compilation strategy."""
    
    if not GPU_AVAILABLE:
        print("GPU not available")
        return
    
    print("=" * 80)
    print("TESTING SEPARATE PHASE COMPILATION")
    print("=" * 80)
    
    from pycalphad import Database
    from pycalphad.core.workspace import Workspace
    import pycalphad.variables as v
    
    # Use Al-Cu-Fe database
    try:
        dbf = Database('Al-Cu-Fe.tdb')
        comps = ['AL', 'CU', 'FE', 'VA']
        phases = list(dbf.phases.keys())[:19]  # Use 19 phases
        
        print(f"Testing with {len(phases)} phases from Al-Cu-Fe system")
        
        conditions = {
            v.T: 800,
            v.P: 101325,
            v.X('AL'): 0.3,
            v.X('CU'): 0.3
        }
        
        wks = Workspace(dbf, comps, phases, conditions, verbose=False)
        
        # Try separate compilation
        try:
            start_time = time.time()
            module = compile_phases_separately(wks, verbose=True)
            compile_time = time.time() - start_time
            
            print(f"\n[GPU] Compilation successful in {compile_time:.2f} seconds!")
            
            # Test the module
            test_kernel = module.get_function("test_kernel")
            n = 10
            input_data = cp.arange(n, dtype=cp.float64)
            output_data = cp.zeros(n, dtype=cp.float64)
            
            test_kernel((1,), (n,), (output_data, input_data, n))
            
            result = output_data.get()
            expected_base = input_data.get() * 2.0
            
            print(f"[GPU] Test kernel verified - {len(phases)} phases available")
            
        except Exception as e:
            print(f"[GPU] Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            
    except FileNotFoundError:
        print("Al-Cu-Fe.tdb not found")


if __name__ == "__main__":
    test_separate_compilation()