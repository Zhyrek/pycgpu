#!/usr/bin/env python
"""
Find which kernel is crashing by intercepting HIP calls
"""

import subprocess
import sys
import os
import re
import time

def parse_hip_trace_for_kernels(stderr_text):
    """Extract kernel launch information from HIP trace"""
    kernels = []
    launch_pattern = r'hipLaunchKernel.*?kernel=(\w+)'
    module_pattern = r'hipModuleLaunchKernel.*?kernel=(\w+)'

    for line in stderr_text.split('\n'):
        # Look for kernel launches
        if 'hipLaunchKernel' in line or 'hipModuleLaunchKernel' in line:
            kernels.append(line.strip())
        # Also catch CuPy kernel launches
        elif 'cuLaunchKernel' in line:
            kernels.append(line.strip())

    return kernels

def run_with_kernel_trace():
    """Trace every kernel launch to find the last one before crash"""
    print("Finding the crashing kernel...")
    print("="*60)

    # Create a wrapper script that prints before each kernel
    wrapper_code = '''
import sys
import os

# Monkey-patch CuPy to trace kernel launches
try:
    import cupy
    original_launch = cupy.RawKernel.__call__

    kernel_counter = [0]

    def traced_launch(self, *args, **kwargs):
        kernel_counter[0] += 1
        kernel_name = getattr(self, 'name', 'unknown')
        print(f"[KERNEL LAUNCH {kernel_counter[0]}] Launching kernel: {kernel_name}", flush=True)
        sys.stderr.flush()
        sys.stdout.flush()

        # Call original
        result = original_launch(self, *args, **kwargs)

        # Force synchronization after each kernel to find exact crash point
        if os.environ.get('FORCE_KERNEL_SYNC') == '1':
            import cupy.cuda
            cupy.cuda.Device().synchronize()
            print(f"[KERNEL LAUNCH {kernel_counter[0]}] Kernel {kernel_name} completed successfully", flush=True)

        return result

    cupy.RawKernel.__call__ = traced_launch
    print("[TRACER] CuPy kernel tracing enabled", flush=True)
except Exception as e:
    print(f"[TRACER] Failed to patch CuPy: {e}", flush=True)

# Now run the actual test
exec(open("test.py").read())
'''

    with open("trace_wrapper.py", "w") as f:
        f.write(wrapper_code)

    # First run: Normal speed, see all kernels launched
    print("\n1. Normal speed trace (see all kernels):")
    print("-"*40)

    env = os.environ.copy()
    env.update({
        "HIP_TRACE_API": "1",
        "HIP_LAUNCH_BLOCKING": "1",
    })

    result = subprocess.run([sys.executable, "trace_wrapper.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    # Find kernel launches
    all_output = result.stdout + "\n" + result.stderr
    kernel_launches = []
    last_kernel = None

    for line in all_output.split('\n'):
        if '[KERNEL LAUNCH' in line:
            kernel_launches.append(line)
            if 'Launching kernel:' in line:
                last_kernel = line

    print(f"Total kernel launches: {len(kernel_launches)}")
    if kernel_launches:
        print("First kernel:", kernel_launches[0] if kernel_launches else "None")
        print("Last kernel before crash:", last_kernel if last_kernel else "None")

    # Second run: With synchronization after each kernel (slower but precise)
    print("\n2. Synchronized trace (find exact crash point):")
    print("-"*40)

    env["FORCE_KERNEL_SYNC"] = "1"

    try:
        result = subprocess.run([sys.executable, "trace_wrapper.py"],
                              env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("Timeout - synchronization is too slow")
        result = subprocess.CompletedProcess(args=[], returncode=1,
                                            stdout="", stderr="Timeout")

    # Find last successful kernel
    successful_kernels = []
    for line in (result.stdout + "\n" + result.stderr).split('\n'):
        if 'completed successfully' in line:
            successful_kernels.append(line)

    if successful_kernels:
        print(f"Last successful kernel: {successful_kernels[-1]}")

        # The crash is in the NEXT kernel after the last successful one
        for i, line in enumerate(kernel_launches):
            if successful_kernels[-1].split(']')[0] in line:
                if i + 1 < len(kernel_launches):
                    print(f"CRASHING KERNEL: {kernel_launches[i+1]}")
                break

    # Clean up
    if os.path.exists("trace_wrapper.py"):
        os.remove("trace_wrapper.py")

    return last_kernel

def inspect_kernel_parameters():
    """Check the parameters being passed to kernels"""
    print("\n3. Checking kernel parameters...")
    print("="*60)

    inspection_code = '''
import sys
import cupy
import numpy as np

# Patch CuPy to inspect kernel parameters
original_launch = cupy.RawKernel.__call__

def inspect_launch(self, grid, block, args, **kwargs):
    kernel_name = getattr(self, 'name', 'unknown')
    print(f"\\n[KERNEL PARAMS] {kernel_name}:")
    print(f"  Grid: {grid}")
    print(f"  Block: {block}")
    print(f"  Args count: {len(args)}")

    # Check for suspicious values
    issues = []

    # Check grid/block dimensions
    if isinstance(grid, tuple):
        for i, g in enumerate(grid):
            if g <= 0:
                issues.append(f"Invalid grid dimension [{i}]={g}")
            if g > 65535:  # Max grid size per dimension
                issues.append(f"Grid dimension [{i}]={g} exceeds maximum")

    if isinstance(block, tuple):
        total_threads = 1
        for i, b in enumerate(block):
            if b <= 0:
                issues.append(f"Invalid block dimension [{i}]={b}")
            total_threads *= b
        if total_threads > 1024:  # Max threads per block
            issues.append(f"Total threads per block {total_threads} exceeds 1024")

    # Check arguments
    for i, arg in enumerate(args[:10]):  # First 10 args only
        if hasattr(arg, 'ptr'):  # CuPy array
            print(f"    Arg[{i}]: CuPy array, ptr={hex(arg.ptr) if arg.ptr else 'NULL'}")
            if arg.ptr == 0:
                issues.append(f"Arg[{i}] is NULL pointer!")
        elif isinstance(arg, (int, float)):
            print(f"    Arg[{i}]: {type(arg).__name__} = {arg}")
            if isinstance(arg, (int, np.integer)) and arg > 1e15:
                issues.append(f"Arg[{i}] suspiciously large: {arg}")

    if issues:
        print(f"  ⚠️ POTENTIAL ISSUES:")
        for issue in issues:
            print(f"    - {issue}")

    return original_launch(self, grid, block, args, **kwargs)

cupy.RawKernel.__call__ = inspect_launch

# Run the test
exec(open("test.py").read())
'''

    with open("inspect_wrapper.py", "w") as f:
        f.write(inspection_code)

    env = os.environ.copy()
    env["HIP_LAUNCH_BLOCKING"] = "1"

    result = subprocess.run([sys.executable, "inspect_wrapper.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    # Look for issues
    for line in result.stdout.split('\n'):
        if 'POTENTIAL ISSUES' in line or '⚠️' in line or 'NULL' in line:
            print(line)

    # Clean up
    if os.path.exists("inspect_wrapper.py"):
        os.remove("inspect_wrapper.py")

def check_memory_at_launch():
    """Check available memory at each kernel launch"""
    print("\n4. Memory availability check...")
    print("="*60)

    memory_check_code = '''
import cupy
import sys

original_launch = cupy.RawKernel.__call__
kernel_count = [0]

def memory_checking_launch(self, *args, **kwargs):
    kernel_count[0] += 1
    kernel_name = getattr(self, 'name', 'unknown')

    # Check memory before launch
    mempool = cupy.get_default_memory_pool()
    device = cupy.cuda.Device()
    free_mem = device.mem_info[0] / (1024**2)  # MB
    total_mem = device.mem_info[1] / (1024**2)  # MB
    used_percent = (1 - free_mem/total_mem) * 100

    print(f"[KERNEL {kernel_count[0]}] {kernel_name}: {free_mem:.1f}MB free ({used_percent:.1f}% used)", flush=True)

    if free_mem < 100:  # Less than 100MB free
        print(f"  ⚠️ WARNING: Low memory before {kernel_name}!", flush=True)

    result = original_launch(self, *args, **kwargs)

    # Check if memory dropped significantly
    new_free = device.mem_info[0] / (1024**2)
    if free_mem - new_free > 500:  # Lost >500MB
        print(f"  Large allocation in {kernel_name}: {free_mem - new_free:.1f}MB", flush=True)

    return result

cupy.RawKernel.__call__ = memory_checking_launch

exec(open("test.py").read())
'''

    with open("memory_wrapper.py", "w") as f:
        f.write(memory_check_code)

    env = os.environ.copy()
    env["HIP_LAUNCH_BLOCKING"] = "1"

    result = subprocess.run([sys.executable, "memory_wrapper.py"],
                          env=env, capture_output=True, text=True, timeout=30)

    print("Memory usage timeline:")
    for line in result.stdout.split('\n'):
        if '[KERNEL' in line or 'WARNING' in line or 'Large allocation' in line:
            print(line)

    # Clean up
    if os.path.exists("memory_wrapper.py"):
        os.remove("memory_wrapper.py")

def main():
    print("AMD GPU Kernel Crash Finder")
    print("============================\n")

    # Run all diagnostics
    last_kernel = run_with_kernel_trace()
    inspect_kernel_parameters()
    check_memory_at_launch()

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

    if last_kernel:
        print(f"The crash occurs in or after: {last_kernel}")
        print("\nNext steps:")
        print("1. Check the kernel's grid/block dimensions")
        print("2. Verify all pointer arguments are valid")
        print("3. Check if kernel uses too much shared memory")
        print("4. Look for array out-of-bounds accesses in that kernel")

if __name__ == "__main__":
    main()