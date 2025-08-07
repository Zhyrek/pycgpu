"""
Modular compilation system for GPU kernels with many phases.
Version 2: Uses include-based approach instead of separate object files.
"""

import os
import re
import hashlib
import tempfile
from typing import List, Tuple, Dict, Optional
import cupy as cp


class ModularGPUCompilerV2:
    """
    Compiler that splits phase functions into separate files when needed,
    but includes them all in a single compilation unit to avoid linking issues.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.max_single_compile_phases = 8  # Threshold for modular compilation
        
    def needs_modular_compilation(self, num_phases: int) -> bool:
        """Check if modular compilation is needed based on phase count."""
        return num_phases > self.max_single_compile_phases
        
    def compile_kernel(self, full_source: str, num_phases: int, dynamic_sizes: dict) -> cp.RawModule:
        """
        Compile the kernel, using modular compilation if needed.
        
        Args:
            full_source: Complete GPU source code
            num_phases: Number of phases in the system
            dynamic_sizes: Dictionary of dynamic kernel sizes for -D flags
            
        Returns:
            Compiled CuPy RawModule
        """
        if not self.needs_modular_compilation(num_phases):
            # Standard compilation
            if self.verbose:
                print(f"[GPU] Using standard compilation for {num_phases} phases")
            return self._standard_compile(full_source, dynamic_sizes)
        else:
            # Modular compilation
            if self.verbose:
                print(f"[GPU] Using modular include-based compilation for {num_phases} phases")
            return self._modular_compile_v2(full_source, dynamic_sizes, num_phases)
            
    def _standard_compile(self, full_source: str, dynamic_sizes: dict) -> cp.RawModule:
        """Standard single-file compilation."""
        # Create -D compiler flags
        define_flags = []
        for define_name, value in dynamic_sizes.items():
            define_flags.append(f'-D{define_name}={value}')
            
        if self.verbose:
            define_flags.append('-DVERBOSE_DEBUG')
            
        compile_options = tuple(['-std=c++11'] + define_flags)
        
        try:
            module = cp.RawModule(code=full_source, options=compile_options, backend='nvcc')
            if self.verbose:
                print("[GPU] Standard compilation successful")
            return module
        except Exception as e:
            if self.verbose:
                print(f"[GPU] Standard compilation failed: {e}")
            raise
            
    def _modular_compile_v2(self, full_source: str, dynamic_sizes: dict, num_phases: int) -> cp.RawModule:
        """
        Modular compilation that splits phase functions into separate files
        but includes them all in a single compilation unit.
        """
        # Extract components from the full source
        phase_functions = self._extract_phase_functions(full_source)
        kernel_code = self._extract_kernel_and_utilities(full_source)
        common_headers = self._extract_common_headers(full_source)
        
        if self.verbose:
            print(f"[GPU] Extracted {len(phase_functions)} phase function groups")
            
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write common headers
            header_file = os.path.join(tmpdir, "common.cuh")
            with open(header_file, 'w') as f:
                f.write(common_headers)
                # Add guards to prevent multiple inclusion
                f.write("\n#ifndef PHASE_FUNCTIONS_INCLUDED\n")
                f.write("#define PHASE_FUNCTIONS_INCLUDED\n\n")
                # Add forward declarations for all phase functions
                f.write(self._generate_forward_declarations(full_source))
                f.write("\n#endif // PHASE_FUNCTIONS_INCLUDED\n")
                
            # Write phase functions to separate files
            phase_files = []
            for i, phase_code in enumerate(phase_functions):
                phase_file = os.path.join(tmpdir, f'phase_group_{i}.cuh')
                with open(phase_file, 'w') as f:
                    # Each phase file just contains the function definitions
                    f.write(f"// Phase function group {i}\n")
                    f.write(phase_code)
                    f.write("\n")
                phase_files.append(f'phase_group_{i}.cuh')
                
            # Create main compilation unit that includes everything
            main_source = self._create_main_compilation_unit(
                tmpdir, header_file, phase_files, kernel_code, num_phases
            )
            
            # Compile the main source
            return self._compile_main_unit(main_source, dynamic_sizes, tmpdir)
            
    def _create_main_compilation_unit(self, tmpdir: str, header_file: str, 
                                     phase_files: List[str], kernel_code: str,
                                     num_phases: int) -> str:
        """
        Create the main compilation unit that includes all phase files.
        """
        main_source = []
        
        # Include common headers
        main_source.append('#include "common.cuh"')
        main_source.append('')
        
        # Include all phase function files
        main_source.append('// Include phase functions')
        for phase_file in phase_files:
            main_source.append(f'#include "{phase_file}"')
        main_source.append('')
        
        # Add the kernel code (with phase functions replaced by their usage)
        main_source.append('// Main kernel and utility functions')
        main_source.append(kernel_code)
        
        return '\n'.join(main_source)
        
    def _compile_main_unit(self, main_source: str, dynamic_sizes: dict, tmpdir: str) -> cp.RawModule:
        """Compile the main unit with all includes."""
        # Write main source to file for compilation
        main_file = os.path.join(tmpdir, 'main.cu')
        with open(main_file, 'w') as f:
            f.write(main_source)
            
        # Read it back for CuPy compilation
        with open(main_file, 'r') as f:
            source_code = f.read()
            
        # Create -D compiler flags
        define_flags = []
        for define_name, value in dynamic_sizes.items():
            define_flags.append(f'-D{define_name}={value}')
            
        if self.verbose:
            define_flags.append('-DVERBOSE_DEBUG')
            
        # Add include path for the temporary directory
        include_flags = [f'-I{tmpdir}']
        
        compile_options = tuple(['-std=c++11'] + define_flags + include_flags)
        
        if self.verbose:
            print(f"[GPU] Compiling main unit with includes from {tmpdir}")
            
        try:
            module = cp.RawModule(code=source_code, options=compile_options, backend='nvcc')
            if self.verbose:
                print("[GPU] Modular include-based compilation successful")
            return module
        except Exception as e:
            if self.verbose:
                print(f"[GPU] Compilation failed: {e}")
                # Save the files for debugging
                import shutil
                debug_dir = '/tmp/gpu_compilation_debug'
                if os.path.exists(debug_dir):
                    shutil.rmtree(debug_dir)
                shutil.copytree(tmpdir, debug_dir)
                print(f"[GPU] Saved compilation files to {debug_dir} for debugging")
            raise
            
    def _extract_phase_functions(self, full_source: str) -> List[str]:
        """
        Extract phase model functions from the full source.
        Groups functions by model to maintain locality.
        """
        functions = []
        lines = full_source.split('\n')
        
        # Find phase model functions (they start with pycgpu_model_)
        current_function = []
        in_function = False
        brace_count = 0
        
        for line in lines:
            if 'pycgpu_model_' in line and '__device__' in line:
                if current_function and brace_count == 0:
                    # Save previous function
                    functions.append('\n'.join(current_function))
                    current_function = []
                in_function = True
                brace_count = 0
                
            if in_function:
                current_function.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and '{' in '\n'.join(current_function):
                    # Function complete
                    functions.append('\n'.join(current_function))
                    current_function = []
                    in_function = False
                    
        # Don't forget the last function
        if current_function:
            functions.append('\n'.join(current_function))
            
        # Group functions by model (typically 9 functions per model)
        # obj, formulaobj, formulagrad, formulahess, internal_cons_func, 
        # internal_cons_jac, mass_obj, formulamole_obj, formulamole_grad
        grouped_functions = []
        functions_per_model = 9
        for i in range(0, len(functions), functions_per_model):
            group = '\n\n'.join(functions[i:i+functions_per_model])
            grouped_functions.append(group)
            
        return grouped_functions
        
    def _extract_kernel_and_utilities(self, full_source: str) -> str:
        """
        Extract everything except phase model functions.
        This includes the kernel, utility functions, and initialization code.
        """
        lines = full_source.split('\n')
        result_lines = []
        
        # Skip phase model functions
        skip_until_line = 0
        for i, line in enumerate(lines):
            if i < skip_until_line:
                continue
                
            if 'pycgpu_model_' in line and '__device__' in line:
                # Skip this function
                brace_count = 0
                for j in range(i, len(lines)):
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    if brace_count == 0 and '{' in '\n'.join(lines[i:j+1]):
                        skip_until_line = j + 1
                        break
            else:
                result_lines.append(line)
                
        return '\n'.join(result_lines)
        
    def _extract_common_headers(self, full_source: str) -> str:
        """Extract headers and structure definitions."""
        lines = full_source.split('\n')
        header_lines = []
        
        # Get everything before the first function
        for line in lines:
            if '__device__' in line or '__global__' in line:
                break
            header_lines.append(line)
            
        return '\n'.join(header_lines)
        
    def _generate_forward_declarations(self, full_source: str) -> str:
        """Generate forward declarations for all phase functions."""
        declarations = []
        
        # Find all phase model function signatures
        pattern = r'__device__\s+(\w+)\s+(pycgpu_model_\d+_\w+)\s*\([^)]*\)'
        matches = re.findall(pattern, full_source)
        
        for return_type, func_name in matches:
            # Determine the parameter list based on return type
            # Scalar functions (return double) have one parameter
            # Array functions (return void) have two parameters
            if return_type == 'double':
                params = '(const double* x)'
            else:  # return_type == 'void'
                params = '(double* out, const double* x)'
                
            declarations.append(f'__device__ {return_type} {func_name}{params};')
            
        return '\n'.join(declarations)