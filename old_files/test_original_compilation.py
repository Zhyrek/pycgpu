#!/usr/bin/env python3
"""
Test with completely original compilation to see if GPU ever worked
"""

import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

clear_cupy_kernel_cache()

# Create a temporary version that uses the absolute original compilation
def create_temp_gpu_equilibrium():
    """Create a temporary version that bypasses dynamic sizing completely"""
    
    import shutil
    original_file = '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_equilibrium.py'
    backup_file = original_file + '.backup'
    
    # Create backup
    shutil.copy2(original_file, backup_file)
    
    # Read current file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Replace the dynamic compilation section with original static compilation
    old_section = '''            # DYNAMIC KERNEL SIZING: Use the sizes computed earlier for cache key
            # This addresses user requirement: "For the GPU hard-coded values like MAX_DOF, the values required 
            # by the kernel should be computed based on the phase records/models in pycalphad, and then passed 
            # to the kernel using the -D flag to define it in the kernel code."
            
            # Create -D compiler flags for dynamic sizing
            define_flags = []
            for define_name, value in dynamic_sizes.items():
                define_flags.append(f'-D{define_name}={value}')
            
            if verbose:
                print(f"[GPU] Using dynamic kernel sizing: {dynamic_sizes}")
                print(f"[GPU] Compiler defines: {define_flags}")
            
            # Compilation options with dynamic defines (must be tuple for CuPy)
            compile_options = tuple(['-std=c++11'] + define_flags)'''
    
    new_section = '''            # ORIGINAL STATIC COMPILATION (for testing)
            if verbose:
                print(f"[GPU] Using original static compilation without -D defines")
            
            # Original compilation options 
            compile_options = ('-std=c++11',)'''
    
    # Replace
    modified_content = content.replace(old_section, new_section)
    
    # Write temporary version
    with open(original_file, 'w') as f:
        f.write(modified_content)
    
    return backup_file

# Create temporary version
backup_file = create_temp_gpu_equilibrium()

try:
    import numpy as np
    import pycalphad as pyc
    from pycalphad import Database, equilibrium, variables as v

    print("🔍 TESTING WITH COMPLETELY ORIGINAL COMPILATION")
    print("=" * 60)
    print("(No -D defines, no dynamic sizing, pure original method)")

    # Test conditions
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }

    print("\n🚀 Attempting GPU calculation with original compilation...")
    result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    print("Result received - checking if it was truly GPU or CPU fallback...")

finally:
    # Restore original file
    import shutil
    shutil.move(backup_file, '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_equilibrium.py')
    print(f"\n✅ Restored original gpu_equilibrium.py from backup")