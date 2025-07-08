#!/usr/bin/env python3
"""
Test NVCC detection for CuPy
"""

import os
import subprocess

print("🔍 **NVCC DETECTION TEST**")
print("=" * 40)

# Check current PATH
print(f"Current PATH: {os.environ.get('PATH', 'NOT SET')}")

# Check if nvcc is in PATH
try:
    result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
    if result.returncode == 0:
        nvcc_path = result.stdout.strip()
        print(f"✅ NVCC found in PATH: {nvcc_path}")
    else:
        print(f"❌ NVCC not found in PATH")
except Exception as e:
    print(f"❌ Error checking NVCC: {e}")

# Check CuPy's NVCC detection
try:
    import cupy
    print(f"✅ CuPy imported successfully")
    
    # Try to access CuPy's NVCC detection
    from cupy.cuda import compiler
    print(f"✅ CuPy compiler module imported")
    
    # Check if CuPy can find NVCC
    if hasattr(compiler, '_nvcc'):
        nvcc_val = compiler._nvcc
        print(f"CuPy _nvcc value: {nvcc_val}")
        if nvcc_val is None:
            print(f"❌ CuPy cannot find NVCC (_nvcc is None)")
        else:
            print(f"✅ CuPy found NVCC: {nvcc_val}")
    else:
        print(f"❌ CuPy compiler module has no _nvcc attribute")
        
except Exception as e:
    print(f"❌ Error checking CuPy NVCC detection: {e}")
    import traceback
    traceback.print_exc()

# Manual PATH fix attempt
conda_env_path = "/home/scott/miniconda3/envs/pycalphad-gpu/bin"
if conda_env_path not in os.environ.get('PATH', ''):
    print(f"\n🔧 Adding conda env to PATH: {conda_env_path}")
    os.environ['PATH'] = conda_env_path + ":" + os.environ.get('PATH', '')
    
    # Re-test NVCC
    try:
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip()
            print(f"✅ NVCC now found: {nvcc_path}")
        else:
            print(f"❌ NVCC still not found after PATH fix")
    except Exception as e:
        print(f"❌ Error re-checking NVCC: {e}")
else:
    print(f"✅ Conda env already in PATH")

print(f"\n📝 NEXT STEPS:")
print(f"1. Ensure NVCC is in PATH before importing CuPy")
print(f"2. Set CUDA_HOME environment variable if needed")
print(f"3. Re-test GPU compilation")