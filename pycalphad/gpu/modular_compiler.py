"""
Modular compilation system for GPU kernels with many phases.
This handles compilation when the generated code becomes too large for single compilation.
"""

import os
import re
import hashlib
import subprocess
import tempfile
from typing import List, Tuple, Dict, Optional
import cupy as cp


class ModularGPUCompiler:
    """
    Compiler that splits phase functions into separate modules when needed.
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
                print(f"[GPU] Using modular compilation for {num_phases} phases")
            return self._modular_compile(full_source, dynamic_sizes)
            
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
            
    def _modular_compile(self, full_source: str, dynamic_sizes: dict) -> cp.RawModule:
        """
        Modular compilation that splits phase functions into separate object files.
        """
        # Extract components from the full source
        phase_functions = self._extract_phase_functions(full_source)
        kernel_code = self._extract_kernel_code(full_source, dynamic_sizes)
        common_headers = self._extract_common_headers(full_source)
        
        if self.verbose:
            print(f"[GPU] Extracted {len(phase_functions)} phase function groups")
            
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write common headers
            header_file = os.path.join(tmpdir, "common.h")
            with open(header_file, 'w') as f:
                f.write(common_headers)
                
            # Compile phase functions into object files
            object_files = []
            for i, phase_code in enumerate(phase_functions):
                obj_file = self._compile_phase_module(
                    phase_code, i, tmpdir, header_file, dynamic_sizes
                )
                object_files.append(obj_file)
                
            # Write main kernel code
            kernel_file = os.path.join(tmpdir, "kernel.cu")
            with open(kernel_file, 'w') as f:
                f.write(kernel_code)
                
            # Link everything together
            return self._link_modules(kernel_file, object_files, tmpdir, dynamic_sizes)
            
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
            if 'pycgpu_model_' in line and ('__device__' in line or 'extern "C"' in line):
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
            
        # Group functions by model (8 functions per model typically)
        grouped_functions = []
        for i in range(0, len(functions), 8):
            group = '\n\n'.join(functions[i:i+8])
            grouped_functions.append(group)
            
        return grouped_functions
        
    def _extract_kernel_code(self, full_source: str, dynamic_sizes: dict) -> str:
        """
        Extract the main kernel and helper functions.
        Replaces f-string expressions with actual values.
        """
        lines = full_source.split('\n')
        kernel_lines = []
        
        # Skip phase model functions, keep everything else
        skip_until_line = 0
        for i, line in enumerate(lines):
            if i < skip_until_line:
                continue
                
            if 'pycgpu_model_' in line and ('__device__' in line):
                # Skip this function
                brace_count = 0
                for j in range(i, len(lines)):
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    if brace_count == 0 and '{' in '\n'.join(lines[i:j+1]):
                        skip_until_line = j + 1
                        break
            else:
                # Replace f-string expressions if present
                line = self._replace_fstring_expressions(line, dynamic_sizes)
                kernel_lines.append(line)
                
        kernel_code = '\n'.join(kernel_lines)
        
        # Add forward declarations for phase functions
        declarations = self._generate_forward_declarations(full_source)
        
        # Find where to insert declarations (after headers, before first function)
        insert_pos = kernel_code.find('__device__')
        if insert_pos > 0:
            kernel_code = kernel_code[:insert_pos] + declarations + '\n\n' + kernel_code[insert_pos:]
            
        return kernel_code
        
    def _replace_fstring_expressions(self, line: str, dynamic_sizes: dict) -> str:
        """Replace f-string style expressions with actual values."""
        # Common f-string patterns in the kernel
        replacements = {
            '{num_unique_models if num_unique_models > 0 else 1}': str(dynamic_sizes.get('MAX_PHASES', 1)),
            '{max_phases}': str(dynamic_sizes.get('MAX_PHASES', 1)),
            '{max_components}': str(dynamic_sizes.get('MAX_COMPONENTS', 4)),
            '{max_dof_per_phase}': str(dynamic_sizes.get('MAX_DOF_PER_PHASE', 4)),
        }
        
        for pattern, replacement in replacements.items():
            line = line.replace(pattern, replacement)
            
        return line
        
    def _extract_common_headers(self, full_source: str) -> str:
        """Extract common headers and structure definitions."""
        lines = full_source.split('\n')
        header_lines = []
        
        for line in lines:
            # Stop at first function definition
            if '__device__' in line or '__global__' in line:
                break
            header_lines.append(line)
            
        # Add structure definitions and constants
        headers = '\n'.join(header_lines)
        
        # Ensure we have the PhaseRecord definition
        if 'struct PhaseRecord' not in headers:
            headers += '\n\n// Forward declaration\nstruct PhaseRecord;\n'
            
        return headers
        
    def _generate_forward_declarations(self, full_source: str) -> str:
        """Generate forward declarations for all phase functions."""
        declarations = []
        
        # Find all phase model function signatures
        pattern = r'__device__\s+(\w+)\s+(pycgpu_model_\d+_\w+)\s*\([^)]*\)'
        matches = re.findall(pattern, full_source)
        
        for return_type, func_name in matches:
            # Determine the parameter list based on function type
            if 'formula' in func_name and 'grad' not in func_name:
                params = '(const double* x)'
            else:
                params = '(double* out, const double* x)'
                
            declarations.append(f'__device__ {return_type} {func_name}{params};')
            
        return '\n'.join(declarations)
        
    def _compile_phase_module(self, phase_code: str, index: int, 
                            tmpdir: str, header_file: str, 
                            dynamic_sizes: dict) -> str:
        """Compile a single phase module to an object file."""
        # Create source file
        source_file = os.path.join(tmpdir, f'phase_{index}.cu')
        with open(source_file, 'w') as f:
            f.write(f'#include "{header_file}"\n\n')
            f.write('extern "C" {\n\n')
            f.write(phase_code)
            f.write('\n\n}  // extern "C"\n')
            
        # Compile to object file
        obj_file = os.path.join(tmpdir, f'phase_{index}.o')
        
        # Build nvcc command
        nvcc_cmd = ['nvcc', '-dc', '-std=c++11']
        
        # Add dynamic defines
        for define_name, value in dynamic_sizes.items():
            nvcc_cmd.extend(['-D', f'{define_name}={value}'])
            
        if self.verbose:
            nvcc_cmd.extend(['-D', 'VERBOSE_DEBUG'])
            
        nvcc_cmd.extend(['-o', obj_file, source_file])
        
        if self.verbose:
            print(f"[GPU] Compiling phase module {index}...")
            
        # Run compilation
        result = subprocess.run(nvcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Phase compilation failed: {result.stderr}")
            
        return obj_file
        
    def _link_modules(self, kernel_file: str, object_files: List[str], 
                      tmpdir: str, dynamic_sizes: dict) -> cp.RawModule:
        """Link all object files and create the final module."""
        # Create a wrapper that includes everything
        wrapper_file = os.path.join(tmpdir, 'wrapper.cu')
        
        with open(wrapper_file, 'w') as f:
            # Read kernel code
            with open(kernel_file, 'r') as kf:
                kernel_code = kf.read()
                
            f.write(kernel_code)
            
        # Now compile wrapper with CuPy, linking the object files
        define_flags = []
        for define_name, value in dynamic_sizes.items():
            define_flags.append(f'-D{define_name}={value}')
            
        if self.verbose:
            define_flags.append('-DVERBOSE_DEBUG')
            
        # Add object files to link
        link_flags = []
        for obj_file in object_files:
            link_flags.append(f'-Xlinker={obj_file}')
            
        compile_options = tuple(['-std=c++11'] + define_flags + link_flags)
        
        if self.verbose:
            print(f"[GPU] Linking {len(object_files)} object files...")
            
        # Read the wrapper file content
        with open(wrapper_file, 'r') as f:
            wrapper_code = f.read()
            
        try:
            module = cp.RawModule(code=wrapper_code, options=compile_options, backend='nvcc')
            if self.verbose:
                print("[GPU] Modular compilation and linking successful")
            return module
        except Exception as e:
            if self.verbose:
                print(f"[GPU] Linking failed: {e}")
            raise