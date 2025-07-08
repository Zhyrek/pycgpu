#!/usr/bin/env python3
"""Test different CuPy compilation options"""

import cupy as cp
import os

# Set environment variable
os.environ['CUPY_NVCC_FLAGS'] = '-allow-unsupported-compiler'

# Simple test kernel in pure C
simple_kernel = r'''
extern "C" __global__
void test_kernel(float* x, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        x[tid] = tid * 2.0f;
    }
}
'''

print("Testing different compilation options...")

# Test 1: Default options
try:
    print("\n1. Testing default compilation...")
    module1 = cp.RawModule(code=simple_kernel)
    print("✓ Default compilation successful")
except Exception as e:
    print(f"✗ Default compilation failed: {e}")

# Test 2: C++11 with allow-unsupported-compiler
try:
    print("\n2. Testing C++11 with -allow-unsupported-compiler...")
    module2 = cp.RawModule(code=simple_kernel, options=('-std=c++11', '-allow-unsupported-compiler'))
    print("✓ C++11 compilation successful")
except Exception as e:
    print(f"✗ C++11 compilation failed: {e}")

# Test 3: C99
try:
    print("\n3. Testing C99...")
    module3 = cp.RawModule(code=simple_kernel, options=('-std=c99', '-allow-unsupported-compiler'))
    print("✓ C99 compilation successful")
except Exception as e:
    print(f"✗ C99 compilation failed: {e}")

# Test 4: No standard specified
try:
    print("\n4. Testing no standard with -allow-unsupported-compiler...")
    module4 = cp.RawModule(code=simple_kernel, options=('-allow-unsupported-compiler',))
    print("✓ No standard compilation successful")
except Exception as e:
    print(f"✗ No standard compilation failed: {e}")

# Test 5: Disable problematic features
try:
    print("\n5. Testing with reduced feature set...")
    options = [
        '-allow-unsupported-compiler',
        '--std=c++11',
        '-D__CUDACC_VER_MAJOR__=12',
        '-D__CUDACC_VER_MINOR__=1'
    ]
    module5 = cp.RawModule(code=simple_kernel, options=options)
    print("✓ Reduced feature compilation successful")
except Exception as e:
    print(f"✗ Reduced feature compilation failed: {e}")

print("\nCompilation test complete!")