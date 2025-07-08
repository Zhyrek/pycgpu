#!/usr/bin/env python3
"""Diagnose CUDA and CuPy setup"""

import sys
import os

print("=== CUDA and CuPy Diagnostics ===")
print(f"Python: {sys.version}")

# Check environment variables
print("\n--- Environment Variables ---")
for var in ['CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")

# Check CuPy
try:
    import cupy
    print("\n--- CuPy Information ---")
    print(f"CuPy version: {cupy.__version__}")
    print(f"CUDA available: {cupy.cuda.is_available()}")
    
    if cupy.cuda.is_available():
        print(f"CUDA runtime version: {cupy.cuda.runtime.runtimeGetVersion()}")
        print(f"CUDA driver version: {cupy.cuda.runtime.driverGetVersion()}")
        
        # Try to get device info
        try:
            device = cupy.cuda.Device()
            print(f"GPU Device: {device.name}")
            print(f"Compute capability: {device.compute_capability}")
        except Exception as e:
            print(f"Error getting device info: {e}")
    
    # Test compilation
    print("\n--- Testing CuPy Compilation ---")
    try:
        # Simple kernel test
        kernel = cupy.RawKernel(r'''
        extern "C" __global__
        void test_kernel(float* x) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            x[tid] = tid;
        }
        ''', 'test_kernel')
        print("✓ Simple kernel compilation successful")
    except Exception as e:
        print(f"✗ Kernel compilation failed: {e}")
        
except ImportError as e:
    print(f"\n✗ CuPy not available: {e}")

# Suggestions
print("\n--- Suggestions ---")
print("If you're seeing 'libnvrtc.so.12' errors, try:")
print("1. Install CUDA Toolkit 12.x from NVIDIA")
print("2. Set LD_LIBRARY_PATH to include CUDA libraries:")
print("   export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH")
print("3. Or use conda/mamba to install cudatoolkit alongside cupy")
print("4. On WSL2, ensure you have the CUDA WSL driver installed on Windows")