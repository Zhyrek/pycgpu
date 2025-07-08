#!/usr/bin/env python3
"""Fix CUDA compiler compatibility issue by setting environment variable"""

import os
import sys

# Set environment variable to allow unsupported compiler
os.environ['CUPY_NVCC_GENERATE_CODE'] = 'arch=compute_89,code=sm_89;arch=compute_75,code=sm_75;arch=compute_70,code=sm_70'
os.environ['CUPY_NVCC_FLAGS'] = '-allow-unsupported-compiler'

# Import after setting environment variables
import cupy as cp

print("Environment configured to allow GCC 13.3 with CUDA")
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")

# Test compilation with the flag
try:
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void test_kernel(float* x) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        x[tid] = tid * 2.0f;
    }
    ''', 'test_kernel')
    print("✓ Kernel compilation successful with -allow-unsupported-compiler flag")
    
    # Test the kernel
    x = cp.zeros(10, dtype=cp.float32)
    kernel((1,), (10,), (x,))
    print(f"✓ Kernel execution successful: {x}")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTo fix the issue permanently, add these lines before importing cupy:")
print("os.environ['CUPY_NVCC_FLAGS'] = '-allow-unsupported-compiler'")