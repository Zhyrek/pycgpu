"""
Multi-file GPU compilation strategy for supporting many phases simultaneously.

This module implements a strategy to compile phase functions separately and link them
together to avoid compilation timeouts and memory issues with large numbers of phases.

Strategy:
1. Generate each phase's device functions (obj, grad, hess, etc.) as a separate .cu file
2. Compile each phase file to a .cubin or .ptx file
3. Link all phase files together with the main kernel
4. Allow the main kernel to call device functions from any phase file

CRITICAL: All phases must be available simultaneously - no shortcuts allowed.
"""

import os
import tempfile
import hashlib
import subprocess
from typing import Dict, List, Tuple, Optional
import cupy as cp
from pathlib import Path

# Import existing codegen functions
from .gpu_codegen import (
    _generate_c_code_for_phase_models,
    _get_c_define,
    _read_gpu_header,
    compute_dynamic_kernel_sizes
)


class MultiFileGPUCompiler:
    """
    Handles multi-file compilation strategy for GPU equilibrium calculations.
    
    This class splits phase functions into separate compilation units to handle
    large numbers of phases without hitting compilation limits.
    """
    
    def __init__(self, workspace, verbose=False):
        self.workspace = workspace
        self.verbose = verbose
        self.temp_dir = None
        self.phase_files = {}  # Maps phase name to file path
        self.compiled_objects = {}  # Maps phase name to compiled object
        
    def _create_temp_directory(self):
        """Create a temporary directory for compilation artifacts."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="pycalphad_gpu_")
            if self.verbose:
                print(f"[GPU] Created temp directory: {self.temp_dir}")
        return self.temp_dir
    
    def _generate_phase_header(self, phase_name: str, phase_idx: int) -> str:
        """
        Generate a header file declaring the device functions for a phase.
        
        This header will be included by other files that need to call these functions.
        """
        header = f"""
#ifndef PHASE_{phase_name.upper()}_H
#define PHASE_{phase_name.upper()}_H

// Forward declarations for phase {phase_name} (index {phase_idx})
extern "C" {{

__device__ double phase_{phase_idx}_obj(const double* x, int x_len);
__device__ void phase_{phase_idx}_grad(double* out, const double* x, int x_len);
__device__ void phase_{phase_idx}_hess(double* out, const double* x, int x_len);
__device__ void phase_{phase_idx}_internal_cons(double* out, const double* x, int x_len);
__device__ void phase_{phase_idx}_internal_cons_jac(double* out, const double* x, int x_len);
__device__ void phase_{phase_idx}_internal_cons_hess(double* out, const double* x, int x_len, int cons_idx);
__device__ void phase_{phase_idx}_masses(double* out, const double* x, int x_len, int num_components);

}}

#endif // PHASE_{phase_name.upper()}_H
"""
        return header
    
    def _generate_phase_source(self, phase_name: str, phase_idx: int, 
                              model_code: str) -> str:
        """
        Generate a source file containing the device functions for a single phase.
        
        Each phase gets its own .cu file to be compiled separately.
        """
        source = f"""
// Phase functions for {phase_name} (index {phase_idx})
#include <math.h>
#include <float.h>

// Include the header with forward declarations
#include "phase_{phase_name}.h"

// Actual function implementations
extern "C" {{

{model_code}

}}
"""
        return source
    
    def _generate_main_kernel_source(self, phase_headers: List[str], 
                                    num_phases: int) -> str:
        """
        Generate the main kernel source that includes all phase headers and
        implements the equilibrium solver.
        """
        # Include all phase headers
        includes = "\n".join([f'#include "{header}"' for header in phase_headers])
        
        # Read static headers
        svd_c_source = _read_gpu_header("svd.c")
        phase_rec_h_source = _read_gpu_header("phase_rec.h")
        comp_set_h_source = _read_gpu_header("comp_set.h")
        lu_solver_h_source = _read_gpu_header("lu_solver.h")
        minimizer_h_source = _read_gpu_header("minimizer.h")
        eqsolver_h_source = _read_gpu_header("eqsolver.h")
        
        source = f"""
// Main equilibrium kernel with all phases linked
#include <math.h>
#include <float.h>
#include <stdio.h>

// Include all phase headers for function declarations
{includes}

// Static includes
{svd_c_source}
{phase_rec_h_source}
{comp_set_h_source}
{lu_solver_h_source}
{minimizer_h_source}
{eqsolver_h_source}

// Global phase records array
__device__ PhaseRecord g_phase_records_array[{num_phases}];

// Function pointer initialization for each phase
__device__ void init_phase_records() {{
    // This function will be called once to set up function pointers
    // for all phases based on the linked phase functions
    
    // Example for phase 0:
    // g_phase_records_array[0].obj = phase_0_obj;
    // g_phase_records_array[0].grad = phase_0_grad;
    // etc.
    
    // This will be generated dynamically based on available phases
}}

extern "C" {{

// Main equilibrium kernel implementation
__global__ void equilibrium_kernel(
    // ... kernel parameters ...
) {{
    // Kernel implementation using the linked phase functions
}}

}}
"""
        return source
    
    def compile_phases_separately(self, phase_models: Dict[str, str]) -> Dict[str, str]:
        """
        Compile each phase's functions as a separate compilation unit.
        
        Args:
            phase_models: Dictionary mapping phase names to their generated C code
            
        Returns:
            Dictionary mapping phase names to compiled object file paths
        """
        self._create_temp_directory()
        compiled_objects = {}
        
        for idx, (phase_name, model_code) in enumerate(phase_models.items()):
            if self.verbose:
                print(f"[GPU] Compiling phase {phase_name} ({idx+1}/{len(phase_models)})...")
            
            # Generate header and source files
            header_content = self._generate_phase_header(phase_name, idx)
            source_content = self._generate_phase_source(phase_name, idx, model_code)
            
            # Write files
            header_path = os.path.join(self.temp_dir, f"phase_{phase_name}.h")
            source_path = os.path.join(self.temp_dir, f"phase_{phase_name}.cu")
            
            with open(header_path, 'w') as f:
                f.write(header_content)
            
            with open(source_path, 'w') as f:
                f.write(source_content)
            
            # Compile to object file using nvcc
            object_path = os.path.join(self.temp_dir, f"phase_{phase_name}.o")
            
            compile_cmd = [
                "nvcc",
                "-c",  # Compile only, don't link
                "-O3",  # Optimization
                "-arch=sm_70",  # Target architecture (adjust as needed)
                "--device-c",  # Compile device code
                "-I", self.temp_dir,  # Include directory for headers
                source_path,
                "-o", object_path
            ]
            
            try:
                result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
                if self.verbose:
                    print(f"[GPU]   Phase {phase_name} compiled successfully")
                compiled_objects[phase_name] = object_path
            except subprocess.CalledProcessError as e:
                print(f"[GPU] ERROR compiling phase {phase_name}:")
                print(f"  Command: {' '.join(compile_cmd)}")
                print(f"  stdout: {e.stdout}")
                print(f"  stderr: {e.stderr}")
                raise
            
        return compiled_objects
    
    def link_phases_to_kernel(self, compiled_objects: Dict[str, str]) -> "cp.RawModule":
        """
        Link all compiled phase objects together with the main kernel.
        
        Args:
            compiled_objects: Dictionary mapping phase names to compiled object paths
            
        Returns:
            CuPy RawModule with all phases linked together
        """
        if self.verbose:
            print(f"[GPU] Linking {len(compiled_objects)} phase objects...")
        
        # Generate main kernel source that references all phases
        phase_headers = [f"phase_{name}.h" for name in compiled_objects.keys()]
        main_source = self._generate_main_kernel_source(phase_headers, len(compiled_objects))
        
        # Write main kernel source
        main_source_path = os.path.join(self.temp_dir, "main_kernel.cu")
        with open(main_source_path, 'w') as f:
            f.write(main_source)
        
        # Link everything together using nvcc
        output_path = os.path.join(self.temp_dir, "equilibrium.ptx")
        
        link_cmd = [
            "nvcc",
            "-ptx",  # Generate PTX
            "-O3",
            "-arch=sm_70",
            "-I", self.temp_dir,
            main_source_path
        ] + list(compiled_objects.values())  # Add all object files
        
        try:
            result = subprocess.run(link_cmd, capture_output=True, text=True, check=True)
            if self.verbose:
                print(f"[GPU] Linking successful, loading module...")
            
            # Load the linked PTX into CuPy
            with open(output_path, 'r') as f:
                ptx_code = f.read()
            
            module = cp.RawModule(code=ptx_code, backend='nvcc')
            
            if self.verbose:
                print(f"[GPU] Module loaded successfully with {len(compiled_objects)} phases")
            
            return module
            
        except subprocess.CalledProcessError as e:
            print(f"[GPU] ERROR linking phases:")
            print(f"  Command: {' '.join(link_cmd)}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            raise
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"[GPU] Cleaned up temp directory: {self.temp_dir}")


def test_multi_file_compilation():
    """Test the multi-file compilation with dummy phase functions."""
    
    # Create dummy phase models
    phase_models = {}
    
    for i in range(21):  # Test with 21 phases
        phase_name = f"PHASE_{i}"
        
        # Generate a simple dummy function for each phase
        model_code = f"""
__device__ double phase_{i}_obj(const double* x, int x_len) {{
    // Dummy objective function for phase {i}
    double sum = 0.0;
    for (int j = 0; j < x_len && j < 10; j++) {{
        sum += x[j] * {i+1}.0;
    }}
    return sum;
}}

__device__ void phase_{i}_grad(double* out, const double* x, int x_len) {{
    // Dummy gradient for phase {i}
    for (int j = 0; j < x_len && j < 10; j++) {{
        out[j] = {i+1}.0;
    }}
}}

__device__ void phase_{i}_hess(double* out, const double* x, int x_len) {{
    // Dummy Hessian for phase {i}
    // Just identity matrix scaled by phase index
    int n = x_len < 10 ? x_len : 10;
    for (int j = 0; j < n*n; j++) {{
        out[j] = (j % (n+1) == 0) ? {i+1}.0 : 0.0;
    }}
}}

__device__ void phase_{i}_internal_cons(double* out, const double* x, int x_len) {{
    // Dummy internal constraints
    out[0] = 0.0;
}}

__device__ void phase_{i}_internal_cons_jac(double* out, const double* x, int x_len) {{
    // Dummy constraint Jacobian
    out[0] = 0.0;
}}

__device__ void phase_{i}_internal_cons_hess(double* out, const double* x, int x_len, int cons_idx) {{
    // Dummy constraint Hessian
    out[0] = 0.0;
}}

__device__ void phase_{i}_masses(double* out, const double* x, int x_len, int num_components) {{
    // Dummy mass calculation
    for (int j = 0; j < num_components && j < 10; j++) {{
        out[j] = x[0] * {i+1}.0;
    }}
}}
"""
        phase_models[phase_name] = model_code
    
    print(f"Testing multi-file compilation with {len(phase_models)} phases...")
    
    # Create compiler instance
    compiler = MultiFileGPUCompiler(None, verbose=True)
    
    try:
        # Compile phases separately
        compiled_objects = compiler.compile_phases_separately(phase_models)
        
        # Link everything together
        module = compiler.link_phases_to_kernel(compiled_objects)
        
        print(f"SUCCESS: Compiled and linked {len(phase_models)} phases!")
        
        # Test that we can get a kernel from the module
        kernel = module.get_function("equilibrium_kernel")
        print(f"Successfully retrieved kernel function: {kernel}")
        
    finally:
        compiler.cleanup()


if __name__ == "__main__":
    # Run test when module is executed directly
    test_multi_file_compilation()