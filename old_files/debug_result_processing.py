#!/usr/bin/env python3
"""
Debug result processing dimension issues
"""

import os
import glob

# Setup CUDA environment BEFORE importing anything
def setup_cuda_environment():
    conda_env_path = "/home/scott/miniconda3/envs/pycalphad-gpu/bin"
    current_path = os.environ.get('PATH', '')
    if conda_env_path not in current_path:
        os.environ['PATH'] = conda_env_path + ":" + current_path
    cuda_home = "/home/scott/miniconda3/envs/pycalphad-gpu"
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['CUDA_ROOT'] = cuda_home

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

setup_cuda_environment()
clear_cupy_kernel_cache()

import numpy as np
import pycalphad as pyc
from pycalphad import Database, variables as v

print("🔍 **DEBUGGING RESULT PROCESSING**")
print("=" * 40)

# Test conditions
tdb = Database("NbTi.tdb")
phases = ["BCC_A2"]
comps = ["NB", "TI", "VA"]
conditions = {
    v.X("TI"): 0.1,
    v.T: 800,
    v.P: 101325
}

try:
    print("🚀 Testing GPU calculation with verbose output...")
    from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
    
    # Call GPU equilibrium with verbose to see result processing details
    result = equilibrium_gpu(tdb, comps, phases, conditions, 
                            gpu=True, verbose=True, fallback_on_error=False)
    
    print("✅ GPU calculation completed!")
    print(f"Result type: {type(result)}")
    if hasattr(result, 'GM'):
        print(f"GM shape: {result.GM.shape}")
        print(f"GM values: {result.GM.values}")

except Exception as e:
    print(f"\n❌ **ERROR IN RESULT PROCESSING:**")
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
    
    import traceback
    print(f"\n🔍 **TRACEBACK:**")
    traceback.print_exc()
    
    # Analyze the error
    error_str = str(e)
    if "dimensions" in error_str and "must have the same length" in error_str:
        print(f"\n🎯 **DIMENSION MISMATCH ANALYSIS:**")
        print("- GPU result array dimensions don't match coordinate system")
        print("- Dynamic sizing working but coordinate handling needs fix")
        print("- Need to ensure result arrays match expected coordinate structure")
    
    print(f"\n📝 **NEXT FIX:** Coordinate system handling in result processing")