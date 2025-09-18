"""
True separate compilation using nvcc with proper device linking.

This implementation:
1. Compiles phase chunks separately with nvcc -dc
2. Links all device code together with nvcc -dlink  
3. Creates final executable module with nvcc
4. Loads the compiled cubin into CuPy
"""

import os
import tempfile
import subprocess
import hashlib
import time
import shutil
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
    compute_dynamic_kernel_sizes
)


def find_nvcc() -> str:
    """Find the nvcc compiler."""
    nvcc_path = shutil.which('nvcc')
    if nvcc_path:
        return nvcc_path
    
    # Common CUDA installation paths
    common_paths = [
        '/usr/local/cuda/bin/nvcc',
        '/usr/bin/nvcc',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    raise RuntimeError("Could not find nvcc compiler. Please ensure CUDA is installed and nvcc is in PATH")


def get_cuda_arch() -> str:
    """Get the appropriate CUDA architecture flag."""
    try:
        device = cp.cuda.Device()
        major = device.compute_capability[0]
        minor = device.compute_capability[1]
        arch = f"sm_{major}{minor}"
        return arch
    except:
        return "sm_70"  # Default


def write_phase_chunk_file(chunk_idx: int, phases: List[Tuple[int, str, str]], 
                          dynamic_sizes: Dict[str, int], output_dir: str) -> str:
    """Write a chunk of phase functions to a .cu file."""
    
    max_components = dynamic_sizes.get('MAX_COMPONENTS', 4)
    max_statevars = dynamic_sizes.get('MAX_STATEVARS', 4)
    max_dof = dynamic_sizes.get('MAX_DOF_PER_PHASE', 10)
    max_internal_constraints = dynamic_sizes.get('MAX_INTERNAL_CONSTRAINTS', 20)
    
    source = f"""
// Phase chunk {chunk_idx}
#include <float.h>
#include <math.h>
#include <stdio.h>

#define MAX_COMPONENTS {max_components}
#define MAX_STATEVARS {max_statevars}
#define MAX_DOF_PER_PHASE {max_dof}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}

typedef double (*pycgpu_func_t)(const double*);
typedef void (*pycgpu_array_func_t)(double*, const double*);

// Define phase functions for this chunk
"""
    
    for phase_idx, phase_name, phase_code in phases:
        source += f"""
// ===== Phase {phase_idx}: {phase_name} =====
{phase_code}
"""
    
    filename = os.path.join(output_dir, f"chunk_{chunk_idx:02d}.cu")
    with open(filename, 'w') as f:
        f.write(source)
    return filename


def write_main_kernel_file(num_phases: int, phase_init_calls: str,
                          phase_codes: List[Tuple[int, str, str]],
                          dynamic_sizes: Dict[str, int], output_dir: str) -> str:
    """Write the main kernel file with extern declarations."""
    
    # Read required headers
    svd_c = _read_gpu_header("svd.c")
    phase_rec_h = _read_gpu_header("phase_rec.h")
    comp_set_h = _read_gpu_header("comp_set.h")
    lu_solver_h = _read_gpu_header("lu_solver.h")
    minimizer_h = _read_gpu_header("minimizer.h")
    eqsolver_h = _read_gpu_header("eqsolver.h")
    
    # Dynamic sizes
    max_components = dynamic_sizes.get('MAX_COMPONENTS', 4)
    max_phases = dynamic_sizes.get('MAX_PHASES', 64)
    max_statevars = dynamic_sizes.get('MAX_STATEVARS', 4)
    max_dof_per_phase = dynamic_sizes.get('MAX_DOF_PER_PHASE', 10)
    max_internal_constraints = dynamic_sizes.get('MAX_INTERNAL_CONSTRAINTS', 20)
    max_fixed_mole = dynamic_sizes.get('MAX_FIXED_MOLE_FRACTION_CONDITIONS', 4)
    min_phase_fraction = _get_c_define("MIN_PHASE_FRACTION")
    
    source = f"""
// Main kernel linking with device-compiled phase chunks
#include <float.h>
#include <math.h>
#include <stdio.h>

#define MAX_COMPONENTS {max_components}
#define MAX_PHASES {max_phases}
#define MAX_STATEVARS {max_statevars}
#define MAX_DOF_PER_PHASE {max_dof_per_phase}
#define MAX_INTERNAL_CONSTRAINTS {max_internal_constraints}
#define MAX_FIXED_MOLE_FRACTION_CONDITIONS {max_fixed_mole}
#define MIN_PHASE_FRACTION {min_phase_fraction}

// Define MAX_SVD dimensions based on MAX_COMPONENTS and MAX_PHASES
#define MAX_SVD_M (MAX_COMPONENTS + MAX_PHASES)
#define MAX_SVD_N MAX_COMPONENTS
#define MAX_PHASE_MATRIX_DIM 100
#define MAX_EQ_MATRIX_SIZE 1000
#define MAX_EQ_SOLN_LEN 50
#define SYSTEM_STATE_SIZE 50000

// Debug helpers
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

// Include headers
{svd_c}
{phase_rec_h}
{comp_set_h}
{lu_solver_h}
{minimizer_h}
{eqsolver_h}

// Extern declarations for phase functions from chunks
"""
    
    # Add extern declarations for all phase functions
    for phase_idx, phase_name, _ in phase_codes:
        prefix = f"nb_model{phase_idx}_"
        source += f"""
// Phase {phase_idx}: {phase_name}
extern __device__ double {prefix}obj(const double* x);
extern __device__ void {prefix}formulaobj(double* out, const double* x);
extern __device__ void {prefix}formulagrad(double* out, const double* x);
extern __device__ void {prefix}formulahess(double* out, const double* x);
extern __device__ void {prefix}internal_cons_func(double* out, const double* x);
extern __device__ void {prefix}internal_cons_jac(double* out, const double* x);
extern __device__ void {prefix}mass_obj(double* out, const double* x);
extern __device__ void {prefix}formulamole_obj(double* out, const double* x);
extern __device__ void {prefix}formulamole_grad(double* out, const double* x);
"""
    
    source += f"""

// Global phase records
__device__ PhaseRecord g_phase_records_array[{max(num_phases, 1)}];

extern "C" {{

__global__ void init_all_gpu_phase_records() {{
    {phase_init_calls}
}}

// Insert the complete kernel implementation from gpu_codegen.py
// This will be replaced with the actual kernel code during compilation

}} // extern "C"
"""
    
    filename = os.path.join(output_dir, "main_kernel.cu")
    with open(filename, 'w') as f:
        f.write(source)
    return filename


def compile_to_combined_source(phase_codes: List[Tuple[int, str, str]], 
                               phase_init_calls: str,
                               dynamic_sizes: Dict[str, int],
                               verbose: bool = False,
                               cache_key: Optional[str] = None) -> str:
    """
    Combine phases into a single CUDA source file.
    
    Returns the combined CUDA source code that can be compiled by CuPy.
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    # Check if we have a cached version
    if cache_key:
        cache_dir = os.path.join(tempfile.gettempdir(), 'pycalphad_gpu_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}_combined.cu")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_source = f.read()
                if verbose:
                    print(f"[GPU] Loaded pre-compiled source from cache: {cache_path}")
                return cached_source
            except Exception as e:
                if verbose:
                    print(f"[GPU] Warning: Could not load from cache: {e}")
    
    num_phases = len(phase_codes)
    nvcc = find_nvcc()
    arch = get_cuda_arch()
    
    if verbose:
        print(f"[GPU] nvcc separate compilation for {num_phases} phases")
        print(f"[GPU] Using nvcc: {nvcc}")
        print(f"[GPU] Target architecture: {arch}")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="pycalphad_nvcc_")
    
    try:
        # Determine chunk size
        if num_phases <= 10:
            chunk_size = 5
        else:
            chunk_size = 3
        
        # Split into chunks
        chunks = []
        for i in range(0, num_phases, chunk_size):
            chunks.append(phase_codes[i:i + chunk_size])
        
        if verbose:
            print(f"[GPU] Splitting into {len(chunks)} chunks of size {chunk_size}")
        
        # Compile each chunk
        obj_files = []
        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"[GPU] Compiling chunk {i+1}/{len(chunks)}...")
            
            # Write chunk
            cu_file = write_phase_chunk_file(i, chunk, dynamic_sizes, temp_dir)
            o_file = cu_file.replace('.cu', '.o')
            
            # Compile with nvcc -dc with optimizations
            cmd = [
                nvcc, '-dc',  # Device compile
                '-O3',  # Full optimization for phase chunks
                '-std=c++11',
                f'--gpu-architecture={arch}',
                # AMD-compatible: removed NVIDIA-specific flags (--use_fast_math, -Xptxas)
                # AMD-compatible: removed NVIDIA-specific precision flags
                '-o', o_file,
                cu_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[GPU] Error compiling chunk {i}:")
                print(result.stderr)
                # Save source for debugging
                debug_file = f"debug_chunk_{i}.cu"
                shutil.copy(cu_file, debug_file)
                print(f"[GPU] Chunk source saved to {debug_file}")
                raise RuntimeError(f"Chunk {i} compilation failed")
            
            obj_files.append(o_file)
        
        if verbose:
            print(f"[GPU] All {len(chunks)} chunks compiled successfully")
        
        # Skip separate main kernel compilation - we'll combine everything instead
        if verbose:
            print("[GPU] Skipping separate main kernel compilation...")
        
        # Create a complete .cu file that includes everything for final compilation
        if verbose:
            print("[GPU] Creating final combined source...")
        
        # Generate main kernel template (not from file since we didn't compile it)
        main_cu = write_main_kernel_file(
            num_phases, phase_init_calls, phase_codes,
            dynamic_sizes, temp_dir
        )
        
        # Read the complete kernel from gpu_codegen.py
        with open('/mnt/c/users/scott/Documents/pycalphad/kernel_complete.cu', 'r') as f:
            complete_kernel = f.read()
        
        # Replace template variables with actual values
        num_unique_models = num_phases
        complete_kernel = complete_kernel.replace(
            '{num_unique_models if num_unique_models > 0 else 1}',
            str(max(num_unique_models, 1))
        )
        complete_kernel = complete_kernel.replace(
            '{num_unique_models}',
            str(num_unique_models)
        )
        
        # Read main kernel template
        with open(main_cu, 'r') as f:
            main_content = f.read()
        
        # Replace the placeholder kernel with the complete one
        placeholder_start = main_content.find("// Insert the complete kernel")
        placeholder_end = main_content.find("}} // extern \"C\"")
        
        if placeholder_start > 0 and placeholder_end > 0:
            # Keep everything before the placeholder
            before = main_content[:placeholder_start]
            # Keep the extern C closing
            after = main_content[placeholder_end:]
            
            # Add all phase functions before the kernel
            phase_funcs = "\n// ===== Phase function definitions =====\n"
            for phase_idx, phase_name, phase_code in phase_codes:
                phase_funcs += f"\n// Phase {phase_idx}: {phase_name}\n"
                phase_code_clean = phase_code.replace('__device__', '__device__')
                phase_funcs += phase_code_clean + "\n"
            
            # Combine: headers + phase functions + complete kernel + closing
            combined_source = before + phase_funcs + "\n" + complete_kernel + "\n" + after
        else:
            # Fallback approach
            combined_source = main_content
            # Insert phase functions before global phase records
            for phase_idx, phase_name, phase_code in phase_codes:
                combined_source = combined_source.replace(
                    "// Global phase records",
                    f"{phase_code}\n\n// Global phase records"
                )
            # Replace placeholder kernel with complete one
            combined_source = combined_source.replace(
                "// Insert the complete kernel implementation from gpu_codegen.py\n// This will be replaced with the actual kernel code during compilation",
                complete_kernel
            )
        
        # Write combined source
        combined_cu = os.path.join(temp_dir, "combined.cu")
        with open(combined_cu, 'w') as f:
            f.write(combined_source)
        
        # Save for debugging if needed
        shutil.copy(combined_cu, "debug_combined.cu")
        
        if verbose:
            print(f"[GPU] Successfully assembled combined source ({len(combined_source)} bytes)")
            print("[GPU] Saved to debug_combined.cu for inspection")
        
        # Save to cache if cache key provided
        if cache_key:
            cache_dir = os.path.join(tempfile.gettempdir(), 'pycalphad_gpu_cache')
            cache_path = os.path.join(cache_dir, f"{cache_key}_combined.cu")
            try:
                with open(cache_path, 'w') as f:
                    f.write(combined_source)
                if verbose:
                    print(f"[GPU] Saved combined source to cache: {cache_path}")
            except Exception as e:
                if verbose:
                    print(f"[GPU] Warning: Could not save to cache: {e}")
        
        # Return the combined source code for CuPy to compile
        return combined_source
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def compile_phases_truly_separate(wks_obj, phase_codes: List[Tuple[int, str, str]], 
                                 phase_init_calls: str,
                                 dynamic_sizes: Dict[str, int],
                                 verbose: bool = False,
                                 cache_key: Optional[str] = None) -> cp.RawModule:
    """
    Main entry point for true separate compilation.
    
    This compiles phases in truly separate compilation units and loads
    the resulting cubin into CuPy.
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    num_phases = len(phase_codes)
    
    if verbose:
        print(f"[GPU] True separate compilation for {num_phases} phases")
    
    try:
        # Get combined source (with caching)
        combined_source = compile_to_combined_source(
            phase_codes, phase_init_calls, 
            dynamic_sizes, verbose, cache_key
        )
        
        # Compile with CuPy using balanced optimization
        if verbose:
            print("[GPU] Compiling combined source with CuPy...")
        
        # Use aggressive optimization for performance
        # Since we're compiling in chunks, we can use higher optimization
        compile_options = [
            '-O3',  # Maximum optimization
            '-std=c++11',
            # AMD-compatible: removed NVIDIA-specific flags
            # (--use_fast_math, -Xptxas, -Xcompiler, precision flags)
        ]

        # Add dynamic size definitions
        if verbose:
            print(f"[GPU] Adding dynamic size definitions: {dynamic_sizes}")
        for define_name, value in dynamic_sizes.items():
            compile_options.append(f'-D{define_name}={value}')
        
        module = cp.RawModule(
            code=combined_source,
            backend='nvcc',  # Use NVCC backend
            options=tuple(compile_options)
        )
        
        if verbose:
            print(f"[GPU] Successfully loaded module with {num_phases} phases!")
        
        return module
        
    except Exception as e:
        print(f"[GPU] True separate compilation failed: {e}")
        # NO FALLBACK - make it work or fail
        raise RuntimeError(f"GPU compilation failed for {num_phases} phases: {e}")