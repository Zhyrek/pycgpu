#!/usr/bin/env python3
"""
Test GPU with proper NVCC environment setup
"""

import os
import glob

# CRITICAL: Set up environment BEFORE importing CuPy
def setup_cuda_environment():
    """Set up CUDA environment variables before importing CuPy"""
    conda_env_path = "/home/scott/miniconda3/envs/pycalphad-gpu/bin"
    
    # Add conda env to PATH if not already there
    current_path = os.environ.get('PATH', '')
    if conda_env_path not in current_path:
        os.environ['PATH'] = conda_env_path + ":" + current_path
        print(f"✅ Added conda env to PATH: {conda_env_path}")
    
    # Set CUDA_HOME if needed
    cuda_home = "/home/scott/miniconda3/envs/pycalphad-gpu"
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['CUDA_ROOT'] = cuda_home
    print(f"✅ Set CUDA_HOME: {cuda_home}")
    
    # Verify NVCC is now accessible
    import subprocess
    try:
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip()
            print(f"✅ NVCC verified: {nvcc_path}")
            return True
        else:
            print(f"❌ NVCC still not found")
            return False
    except Exception as e:
        print(f"❌ Error verifying NVCC: {e}")
        return False

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
                print(f"Removed cached kernel: {cubin_file}")
            except OSError:
                pass

print("🚀 **GPU TEST WITH NVCC FIX**")
print("=" * 40)

# Step 1: Setup environment
if not setup_cuda_environment():
    print("❌ Failed to setup CUDA environment")
    exit(1)

# Step 2: Clear cache
clear_cupy_kernel_cache()

# Step 3: Now import CuPy and pycalphad
try:
    import cupy as cp
    print(f"✅ CuPy imported successfully after environment setup")
    
    # Test if CuPy can find NVCC now
    from cupy.cuda import compiler
    if hasattr(compiler, '_nvcc'):
        nvcc_val = compiler._nvcc
        print(f"CuPy _nvcc value: {nvcc_val}")
    
except Exception as e:
    print(f"❌ CuPy import failed: {e}")
    exit(1)

# Step 4: Import pycalphad and test
import numpy as np
import pycalphad as pyc
from pycalphad import Database, variables as v

# Test conditions
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

print(f"\nTest conditions: {conditions}")

try:
    print("\n🚀 Attempting GPU calculation with NVCC fix...")
    from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
    
    result = equilibrium_gpu(tdb, comps, phases, conditions, 
                            gpu=True, verbose=True, fallback_on_error=False)
    
    print("🎉 **GPU SUCCESS!** Compilation worked!")
    print(f"GM: {result.GM.values.flatten()[0]:.6f} J/mol")
    
except Exception as e:
    print(f"\n❌ **GPU STILL FAILED:**")
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
    
    import traceback
    print(f"\n🔍 **TRACEBACK:**")
    traceback.print_exc()
    
    # Analyze the error
    error_str = str(e)
    if "split" in error_str:
        print(f"\n🎯 Still a 'split' error - may need deeper NVCC setup")
    elif "nvcc" in error_str.lower():
        print(f"\n🎯 NVCC-related error - environment setup incomplete")
    else:
        print(f"\n🎯 Different error - progress made, new issue to fix")