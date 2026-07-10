#!/usr/bin/env python
"""
Diagnose memory access fault in AMD GPU kernel
"""

import subprocess
import sys
import os
import re

def test_with_bounds_checking():
    """Enable GPU memory bounds checking"""
    print("1. Testing with memory bounds checking...")
    print("="*60)

    env = os.environ.copy()
    env.update({
        "HIP_LAUNCH_BLOCKING": "1",
        "AMD_SERIALIZE_KERNEL": "3",
        "HSA_ENABLE_INTERRUPT": "0",  # Disable interrupts for clearer errors
        "HSA_ENABLE_DEBUG": "1",
        "AMD_LOG_LEVEL": "4",
        "GPU_DUMP_DEVICE_KERNEL": "3",  # Dump kernel code
        "HSA_TOOLS_LIB": "librocm-debug-agent.so.2",  # Memory debug agent if available
    })

    result = subprocess.run([sys.executable, "test.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    # Look for specific memory errors
    memory_errors = []
    for line in result.stderr.split('\n'):
        if any(x in line.lower() for x in ['memory', 'fault', 'access', 'violation', 'bounds', 'overflow']):
            memory_errors.append(line)

    if memory_errors:
        print("Memory-related errors found:")
        for err in memory_errors[:10]:
            print(f"  {err}")

    return result.returncode

def test_thread_configs():
    """Test with different thread configurations to isolate the issue"""
    print("\n2. Testing different thread configurations...")
    print("="*60)

    # Create test wrapper that modifies kernel launch parameters
    wrapper_code = '''
import cupy
import sys

# Patch to try single thread execution
original_launch = cupy.RawKernel.__call__

def modified_launch(self, grid, block, args, **kwargs):
    kernel_name = getattr(self, 'name', 'unknown')

    # Force single thread to isolate memory issues
    if "FORCE_SINGLE_THREAD" in os.environ:
        print(f"[DEBUG] Forcing single thread for {kernel_name}", flush=True)
        grid = (1, 1, 1)
        block = (1, 1, 1)
    elif "LIMIT_THREADS" in os.environ:
        limit = int(os.environ["LIMIT_THREADS"])
        if isinstance(block, tuple) and block[0] > limit:
            print(f"[DEBUG] Limiting threads from {block[0]} to {limit} for {kernel_name}", flush=True)
            block = (limit, block[1], block[2])

    print(f"[LAUNCH] {kernel_name}: grid={grid}, block={block}", flush=True)
    return original_launch(self, grid, block, args, **kwargs)

cupy.RawKernel.__call__ = modified_launch

import os
exec(open("test.py").read())
'''

    with open("thread_wrapper.py", "w") as f:
        f.write(wrapper_code)

    configs = [
        ("Single thread", {"FORCE_SINGLE_THREAD": "1"}),
        ("32 threads", {"LIMIT_THREADS": "32"}),
        ("64 threads", {"LIMIT_THREADS": "64"}),
        ("128 threads", {"LIMIT_THREADS": "128"}),
        ("256 threads (original)", {}),
    ]

    for name, env_vars in configs:
        print(f"\nTesting: {name}")
        env = os.environ.copy()
        env["HIP_LAUNCH_BLOCKING"] = "1"
        env.update(env_vars)

        try:
            result = subprocess.run([sys.executable, "thread_wrapper.py"],
                                  env=env, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"  ✓ SUCCESS with {name}")
                # Find what makes it work
                if "FORCE_SINGLE_THREAD" in env_vars:
                    print("  → Issue is thread-related (race condition or shared memory)")
                elif "LIMIT_THREADS" in env_vars:
                    print(f"  → Issue appears above {env_vars['LIMIT_THREADS']} threads")
            else:
                print(f"  ✗ FAILED with {name}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT with {name}")

    if os.path.exists("thread_wrapper.py"):
        os.remove("thread_wrapper.py")

def test_array_access_pattern():
    """Check if it's an array indexing issue"""
    print("\n3. Checking for array access issues...")
    print("="*60)

    # Add array bounds checking
    check_code = '''
import cupy
import numpy as np
import sys

original_launch = cupy.RawKernel.__call__

def check_arrays(self, grid, block, args, **kwargs):
    kernel_name = getattr(self, 'name', 'unknown')

    # Calculate total threads
    total_threads = 1
    if isinstance(grid, tuple):
        for g in grid:
            total_threads *= g
    if isinstance(block, tuple):
        for b in block:
            total_threads *= b

    print(f"[BOUNDS] {kernel_name}: Total threads = {total_threads}", flush=True)

    # Check array sizes vs thread count
    for i, arg in enumerate(args):
        if hasattr(arg, 'size'):  # CuPy array
            if arg.size > 0:
                elements_per_thread = arg.size / total_threads
                print(f"  Arg[{i}]: {arg.size} elements, {elements_per_thread:.2f} per thread", flush=True)

                # Common issue: array too small for thread count
                if arg.size < total_threads:
                    print(f"  ⚠️ WARNING: Array smaller than thread count!", flush=True)

                # Check for suspicious sizes (potential off-by-one)
                if arg.size == total_threads - 1:
                    print(f"  ⚠️ WARNING: Array size is threads-1 (off-by-one error?)", flush=True)

    return original_launch(self, grid, block, args, **kwargs)

cupy.RawKernel.__call__ = check_arrays

import os
exec(open("test.py").read())
'''

    with open("bounds_wrapper.py", "w") as f:
        f.write(check_code)

    env = os.environ.copy()
    env["HIP_LAUNCH_BLOCKING"] = "1"

    result = subprocess.run([sys.executable, "bounds_wrapper.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    # Look for warnings
    for line in result.stdout.split('\n'):
        if '⚠️' in line or 'WARNING' in line or '[BOUNDS]' in line:
            print(line)

    if os.path.exists("bounds_wrapper.py"):
        os.remove("bounds_wrapper.py")

def test_null_pointers():
    """Check for NULL pointer dereferences"""
    print("\n4. Checking for NULL pointers...")
    print("="*60)

    null_check_code = '''
import cupy
import sys

original_launch = cupy.RawKernel.__call__

def null_check(self, grid, block, args, **kwargs):
    kernel_name = getattr(self, 'name', 'unknown')

    null_found = False
    for i, arg in enumerate(args):
        if hasattr(arg, 'ptr'):  # CuPy array
            if arg.ptr == 0:
                print(f"  ❌ NULL POINTER: Arg[{i}] in {kernel_name}!", flush=True)
                null_found = True
            elif arg.size == 0:
                print(f"  ⚠️ EMPTY ARRAY: Arg[{i}] has size 0 in {kernel_name}", flush=True)
        elif arg is None:
            print(f"  ❌ NULL ARGUMENT: Arg[{i}] is None in {kernel_name}!", flush=True)
            null_found = True

    if null_found:
        print(f"[ERROR] Kernel {kernel_name} has NULL arguments - will crash!", flush=True)
        # Could raise exception here to prevent crash
        # raise RuntimeError(f"NULL pointer detected in {kernel_name}")

    return original_launch(self, grid, block, args, **kwargs)

cupy.RawKernel.__call__ = null_check

exec(open("test.py").read())
'''

    with open("null_wrapper.py", "w") as f:
        f.write(null_check_code)

    env = os.environ.copy()
    env["HIP_LAUNCH_BLOCKING"] = "1"

    result = subprocess.run([sys.executable, "null_wrapper.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    # Look for NULL issues
    null_errors = False
    for line in result.stdout.split('\n'):
        if '❌' in line or 'NULL' in line:
            print(line)
            null_errors = True

    if null_errors:
        print("\n⚠️ NULL pointer issue detected - this is likely the cause!")

    if os.path.exists("null_wrapper.py"):
        os.remove("null_wrapper.py")

    return null_errors

def main():
    print("AMD GPU Memory Fault Diagnosis")
    print("==============================\n")

    # Run diagnostics
    test_with_bounds_checking()
    test_thread_configs()
    test_array_access_pattern()
    null_found = test_null_pointers()

    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)

    print("\nThe memory fault is likely due to:")
    if null_found:
        print("• NULL pointer dereference (check array allocations)")
    print("• Array index out of bounds (thread_idx >= array_size)")
    print("• Stack array overflow within kernel")
    print("• Uninitialized memory access")

    print("\nTo fix:")
    print("1. Check all array allocations are successful")
    print("2. Verify array sizes match thread counts")
    print("3. Look for thread_idx based array accesses")
    print("4. Check for stack-allocated arrays in the kernel")

if __name__ == "__main__":
    main()