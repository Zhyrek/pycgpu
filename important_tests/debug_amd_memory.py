#!/usr/bin/env python
"""
AMD GPU Memory Debugging Script
Runs tests with various debugging environment variables
"""
import os
import sys
import subprocess
import time
import cupy as cp

def run_with_env(env_vars, description):
    """Run test with specific environment variables"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Environment: {env_vars}")
    print(f"{'='*60}")

    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)

    # Run the test
    cmd = [sys.executable, "important_tests/test_all_phases.py", "--alcufe-only"]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("TIMEOUT after 30 seconds")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    try:
        mempool = cp.get_default_memory_pool()
        print(f"CuPy memory pool used: {mempool.used_bytes() / 1024**2:.2f} MB")
        print(f"CuPy memory pool total: {mempool.total_bytes() / 1024**2:.2f} MB")

        # Get device properties
        device = cp.cuda.Device()
        free_mem = device.mem_info[0] / 1024**2
        total_mem = device.mem_info[1] / 1024**2
        print(f"GPU free memory: {free_mem:.2f} MB")
        print(f"GPU total memory: {total_mem:.2f} MB")
    except:
        print("Could not query GPU memory")

def main():
    print("AMD GPU Memory Debugging Tool")
    print("==============================")

    # Check initial GPU state
    check_gpu_memory()

    # Test 1: Enable HIP debugging
    run_with_env({
        "AMD_LOG_LEVEL": "4",  # Maximum verbosity
        "HIP_VISIBLE_DEVICES": "0",
        "HIP_LAUNCH_BLOCKING": "1",  # Synchronous kernel launches
        "ROCM_VISIBLE_DEVICES": "0"
    }, "HIP Debug Mode - Maximum Verbosity")

    # Test 2: Track memory allocations
    run_with_env({
        "HSA_ENABLE_SDMA": "0",  # Disable DMA to simplify debugging
        "HIP_TRACE_API": "1",  # Trace all HIP API calls
        "HIP_TRACE_ACTIVITY": "1",
        "AMD_DIRECT_DISPATCH": "0"  # Disable direct dispatch
    }, "Memory Allocation Tracking")

    # Test 3: Force smaller stack size
    run_with_env({
        "HIP_STACK_SIZE": "8192",  # Try different stack sizes (in bytes)
        "HIP_LAUNCH_BLOCKING": "1"
    }, "Reduced Stack Size (8KB)")

    # Test 4: Debug kernel launches
    run_with_env({
        "HIP_DB": "1",  # Enable HIP debugger
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_MAX_HW_QUEUES": "1"  # Single queue for easier debugging
    }, "Kernel Launch Debugging")

    # Test 5: ROCm debug mode
    run_with_env({
        "ROCR_VISIBLE_DEVICES": "0",
        "HSA_ENABLE_DEBUG": "1",
        "HSA_TOOLS_LIB": "libroctracer64.so",  # Enable tracing if available
        "ROCM_DEBUG_ENABLE": "1"
    }, "ROCm Debug Mode")

    # Test 6: Memory pool settings
    run_with_env({
        "HIP_HIDDEN_FREE_MEM": "0",  # Don't hide free memory
        "GPU_MAX_HEAP_SIZE": "50",  # Limit heap to 50% of GPU memory
        "GPU_SINGLE_ALLOC_PERCENT": "50"  # Limit single allocation size
    }, "Memory Pool Restrictions")

    # Final memory check
    print("\nFinal GPU state:")
    check_gpu_memory()

if __name__ == "__main__":
    main()