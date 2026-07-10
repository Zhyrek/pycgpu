#!/usr/bin/env python
"""
Test if HIP correctly receives and uses macro definitions
"""

import cupy as cp
import numpy as np

# Test 1: Simple macro test
print("Test 1: Basic macro passing")
print("-" * 40)

test_kernel_code = '''
extern "C" __global__ void test_defines(double* output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        // These should be defined via -D flags
        output[0] = MAX_COMPONENTS;
        output[1] = MAX_PHASES;
        output[2] = MAX_DOF_PER_PHASE;
        output[3] = MAX_STATEVARS;

        // Print to verify (will show in stderr with verbose)
        printf("GPU: MAX_COMPONENTS=%d\\n", MAX_COMPONENTS);
        printf("GPU: MAX_PHASES=%d\\n", MAX_PHASES);
        printf("GPU: MAX_DOF_PER_PHASE=%d\\n", MAX_DOF_PER_PHASE);
        printf("GPU: MAX_STATEVARS=%d\\n", MAX_STATEVARS);
    }
}
'''

# Define values
defines = {
    'MAX_COMPONENTS': 4,
    'MAX_PHASES': 21,
    'MAX_DOF_PER_PHASE': 16,
    'MAX_STATEVARS': 2
}

# Create compile options
compile_options = ['-std=c++11']
for name, value in defines.items():
    compile_options.append(f'-D{name}={value}')

print(f"Compile options: {compile_options}")

# Compile and run
try:
    module = cp.RawModule(code=test_kernel_code, options=tuple(compile_options))
    kernel = module.get_function('test_defines')

    # Allocate output
    output = cp.zeros(4, dtype=cp.float64)

    # Launch kernel
    kernel((1,), (1,), (output,))

    # Get results
    result = output.get()
    print(f"\nValues received in kernel:")
    print(f"  MAX_COMPONENTS = {int(result[0])}")
    print(f"  MAX_PHASES = {int(result[1])}")
    print(f"  MAX_DOF_PER_PHASE = {int(result[2])}")
    print(f"  MAX_STATEVARS = {int(result[3])}")

    # Verify
    if (int(result[0]) == defines['MAX_COMPONENTS'] and
        int(result[1]) == defines['MAX_PHASES']):
        print("✓ Macros passed correctly!")
    else:
        print("✗ Macros NOT passed correctly!")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Check what happens without defines
print("\n\nTest 2: Kernel without -D flags (should fail)")
print("-" * 40)

test_kernel_no_defines = '''
extern "C" __global__ void test_no_defines(double* output) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        // Try to use undefined macros - should cause compilation error
        #ifdef MAX_COMPONENTS
            output[0] = MAX_COMPONENTS;
        #else
            output[0] = -999;  // Marker for undefined
        #endif

        #ifdef MAX_PHASES
            output[1] = MAX_PHASES;
        #else
            output[1] = -999;
        #endif
    }
}
'''

try:
    # Compile WITHOUT defines
    module2 = cp.RawModule(code=test_kernel_no_defines, options=('-std=c++11',))
    kernel2 = module2.get_function('test_no_defines')

    output2 = cp.zeros(2, dtype=cp.float64)
    kernel2((1,), (1,), (output2,))

    result2 = output2.get()
    if result2[0] == -999:
        print("✓ Correctly detected missing defines")
    else:
        print(f"✗ Unexpected: Got MAX_COMPONENTS={result2[0]} without defining it!")

except Exception as e:
    print(f"Compilation/execution failed as expected: {str(e)[:100]}")

# Test 3: Check memory allocation based on defines
print("\n\nTest 3: Memory allocation using macros")
print("-" * 40)

allocation_test = '''
extern "C" __global__ void test_allocation(double* output, double* work_array) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Calculate expected array size
    const int EXPECTED_SIZE = MAX_PHASES * MAX_COMPONENTS;

    if (tid == 0) {
        output[0] = EXPECTED_SIZE;

        // Try to write to the array at various offsets
        work_array[0] = 1.0;  // Should work
        work_array[EXPECTED_SIZE/2] = 2.0;  // Should work
        work_array[EXPECTED_SIZE-1] = 3.0;  // Should work at boundary

        // DON'T test beyond boundary - would crash
        // work_array[EXPECTED_SIZE] = 4.0;  // Would crash

        printf("GPU: Array size = %d elements\\n", EXPECTED_SIZE);
    }
}
'''

try:
    module3 = cp.RawModule(code=allocation_test, options=tuple(compile_options))
    kernel3 = module3.get_function('test_allocation')

    # Allocate based on the defines we passed
    expected_size = defines['MAX_PHASES'] * defines['MAX_COMPONENTS']
    work_array = cp.zeros(expected_size, dtype=cp.float64)
    output3 = cp.zeros(1, dtype=cp.float64)

    kernel3((1,), (1,), (output3, work_array))

    kernel_size = int(output3.get()[0])
    print(f"Kernel calculated size: {kernel_size}")
    print(f"Python calculated size: {expected_size}")

    if kernel_size == expected_size:
        print("✓ Kernel and Python agree on sizes!")
    else:
        print("✗ SIZE MISMATCH - macros may not be working!")

    # Check if writes succeeded
    work_result = work_array.get()
    if work_result[0] == 1.0 and work_result[expected_size-1] == 3.0:
        print("✓ Array writes successful")
    else:
        print("✗ Array writes failed")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: Check if backend matters
print("\n\nTest 4: Backend check")
print("-" * 40)

import os
if 'hip' in cp.cuda.runtime.runtimeGetVersion().__str__().lower():
    print("Using HIP backend")
else:
    print("Using CUDA backend")

# For HIP, check if different syntax is needed
if os.path.exists('/opt/rocm'):
    print("ROCm detected - using HIP")
    # HIP might need different compilation flags
    hip_options = ['-std=c++14']  # HIP often needs C++14
    for name, value in defines.items():
        hip_options.append(f'-D{name}={value}')

    try:
        module_hip = cp.RawModule(code=test_kernel_code, options=tuple(hip_options))
        print("✓ HIP compilation successful with C++14")
    except:
        print("✗ HIP compilation failed even with C++14")

print("\n" + "="*60)
print("SUMMARY:")
print("If macros aren't being passed correctly to HIP, the kernel will")
print("use default/undefined values, potentially causing huge allocations.")
print("Check the 'GPU:' printf output above to see actual values in kernel.")