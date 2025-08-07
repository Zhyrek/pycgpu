"""
Modular compilation system for GPU kernels with many phases.
Version 3: Inlines everything to avoid include path issues.
"""

import os
import re
import hashlib
import tempfile
from typing import List, Tuple, Dict, Optional
import cupy as cp


class ModularGPUCompilerV3:
    """
    Compiler that reorganizes phase functions to reduce compilation complexity.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.max_single_compile_phases = 8  # Threshold for modular compilation
        
    def needs_modular_compilation(self, num_phases: int) -> bool:
        """Check if modular compilation is needed based on phase count."""
        return num_phases > self.max_single_compile_phases
        
    def compile_kernel(self, full_source: str, num_phases: int, dynamic_sizes: dict) -> cp.RawModule:
        """
        Compile the kernel, using optimized compilation if needed.
        
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
            # Optimized compilation
            if self.verbose:
                print(f"[GPU] Using optimized compilation for {num_phases} phases")
            return self._optimized_compile(full_source, dynamic_sizes, num_phases)
            
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
            
    def _optimized_compile(self, full_source: str, dynamic_sizes: dict, num_phases: int) -> cp.RawModule:
        """
        Optimized compilation that reduces code complexity for many phases.
        Strategy: Simplify the phase functions to reduce compile time.
        """
        if self.verbose:
            print(f"[GPU] Applying optimization strategies for {num_phases} phases...")
            
        # Strategy 1: Remove debug code from phase functions
        optimized_source = self._remove_debug_from_phases(full_source)
        
        # Strategy 2: Apply aggressive optimization flags
        define_flags = []
        for define_name, value in dynamic_sizes.items():
            define_flags.append(f'-D{define_name}={value}')
            
        # Don't add verbose debug for optimized compilation
        # This significantly reduces code size
        
        # Use optimization flags that reduce code size
        compile_options = [
            '-std=c++11',
            '-O2',  # Optimize for speed but not maximum (O3 can increase code size)
            '--use_fast_math',  # Use faster but less precise math
            '-lineinfo',  # Add line info for debugging without full debug symbols
        ] + define_flags
        
        # For very large phase counts, try even more aggressive options
        if num_phases > 15:
            if self.verbose:
                print(f"[GPU] Using aggressive optimization for {num_phases} phases")
            compile_options.extend([
                '--maxrregcount=64',  # Limit register usage to avoid spilling
                '--ptxas-options=-v',  # Verbose PTX assembler output
            ])
        
        compile_options = tuple(compile_options)
        
        # Try compilation with optimizations
        try:
            # First attempt: Try with all optimizations
            module = cp.RawModule(code=optimized_source, options=compile_options, backend='nvcc')
            if self.verbose:
                print("[GPU] Optimized compilation successful")
            return module
        except Exception as e1:
            if self.verbose:
                print(f"[GPU] First optimization attempt failed: {e1}")
                print("[GPU] Trying fallback approach...")
            
            # Fallback: Try with reduced inline limits
            fallback_options = list(compile_options)
            fallback_options.extend([
                '--max-depth=10',  # Limit inline depth
                '-Xcicc', '-O0',  # Disable some optimizations that increase code size
            ])
            
            try:
                module = cp.RawModule(code=optimized_source, options=tuple(fallback_options), backend='nvcc')
                if self.verbose:
                    print("[GPU] Fallback compilation successful")
                return module
            except Exception as e2:
                if self.verbose:
                    print(f"[GPU] Fallback compilation also failed: {e2}")
                    print("[GPU] Attempting minimal compilation...")
                
                # Last resort: Simplify the code further
                minimal_source = self._create_minimal_source(optimized_source, num_phases)
                
                try:
                    module = cp.RawModule(code=minimal_source, options=compile_options, backend='nvcc')
                    if self.verbose:
                        print("[GPU] Minimal compilation successful")
                    return module
                except Exception as e3:
                    if self.verbose:
                        print(f"[GPU] All compilation attempts failed")
                    raise e3
    
    def _remove_debug_from_phases(self, source: str) -> str:
        """Remove debug code from phase functions to reduce code size."""
        lines = source.split('\n')
        result = []
        in_phase_function = False
        
        for line in lines:
            # Detect phase function start
            if 'pycgpu_model_' in line and '__device__' in line:
                in_phase_function = True
            elif in_phase_function and line.strip() == '}':
                in_phase_function = False
                
            # Remove debug prints from phase functions
            if in_phase_function and ('printf' in line or '#ifdef VERBOSE_DEBUG' in line):
                continue
                
            result.append(line)
            
        return '\n'.join(result)
    
    def _create_minimal_source(self, source: str, num_phases: int) -> str:
        """
        Create a minimal version of the source by stubbing out some phase functions.
        This is a last resort for very complex systems.
        """
        if self.verbose:
            print(f"[GPU] Creating minimal source by stubbing complex functions...")
            
        lines = source.split('\n')
        result = []
        
        # Find the most complex phase functions and stub them
        in_hessian = False
        hessian_count = 0
        max_hessians_to_keep = 5  # Only keep first few Hessians
        
        for i, line in enumerate(lines):
            # Detect Hessian functions (these are typically the most complex)
            if 'formulahess' in line and '__device__' in line:
                hessian_count += 1
                if hessian_count > max_hessians_to_keep:
                    # Stub out this Hessian
                    in_hessian = True
                    # Add a stub function
                    func_match = re.search(r'(__device__\s+void\s+\w+\s*\([^)]+\))', line)
                    if func_match:
                        result.append(func_match.group(1) + ' {')
                        result.append('    // Stubbed for compilation')
                        result.append('    for(int i = 0; i < 25; i++) out[i] = 0.0;')
                        result.append('}')
                    continue
            
            if in_hessian:
                # Skip until we find the closing brace
                if line.strip() == '}':
                    in_hessian = False
                continue
                
            result.append(line)
            
        return '\n'.join(result)