#!/usr/bin/env python
"""
Lightweight AMD GPU debugging - finds crash location without massive slowdown
"""

import subprocess
import sys
import os
import time

def run_with_basic_check():
    """Just get the error message quickly"""
    print("1. Quick error check (normal speed)...")
    print("="*60)

    env = os.environ.copy()
    env.update({
        "HIP_LAUNCH_BLOCKING": "1",  # Synchronous but not too slow
        "AMD_LOG_LEVEL": "2",  # Moderate logging
    })

    start = time.time()
    result = subprocess.run([sys.executable, "test.py"], env=env, capture_output=True, text=True)
    elapsed = time.time() - start

    print(f"Runtime: {elapsed:.2f} seconds")
    if result.returncode != 0:
        print("CRASHED! Last 20 lines of stderr:")
        lines = result.stderr.split('\n')
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
    else:
        print("No crash detected")

    return result.returncode != 0

def run_with_hip_check():
    """Add HIP error checking - slightly slower"""
    print("\n2. HIP error location check (2-3x slower)...")
    print("="*60)

    env = os.environ.copy()
    env.update({
        "HIP_LAUNCH_BLOCKING": "1",
        "AMD_SERIALIZE_KERNEL": "1",  # Level 1 is faster than 3
        "HIP_CHECK_ERRORS": "1",
        "AMD_LOG_LEVEL": "3",
    })

    start = time.time()
    result = subprocess.run([sys.executable, "test.py"], env=env, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - start

    print(f"Runtime: {elapsed:.2f} seconds")

    # Look for HIP errors
    for line in result.stderr.split('\n'):
        if 'error' in line.lower() or 'hip' in line.lower() or 'memory' in line.lower():
            print(f"ERROR: {line}")

    return result.returncode

def run_with_stack_check():
    """Check for stack overflow specifically"""
    print("\n3. Stack overflow check (minimal overhead)...")
    print("="*60)

    # Try with different stack sizes to see if it's a stack issue
    stack_sizes = ["4096", "8192", "16384", "32768"]

    for stack_size in stack_sizes:
        print(f"\nTrying stack size: {stack_size} bytes")
        env = os.environ.copy()
        env.update({
            "HIP_STACK_SIZE": stack_size,
            "HIP_LAUNCH_BLOCKING": "1",
        })

        try:
            result = subprocess.run([sys.executable, "test.py"], env=env,
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"  SUCCESS with stack size {stack_size}!")
                return True
            else:
                print(f"  Failed with stack size {stack_size}")
        except subprocess.TimeoutExpired:
            print(f"  Timeout with stack size {stack_size}")

    return False

def run_with_memory_limit():
    """Limit memory to see if it's allocation size"""
    print("\n4. Memory allocation limit test...")
    print("="*60)

    limits = ["10", "25", "50", "75", "90"]  # Percentage of GPU memory

    for limit in limits:
        print(f"\nTrying with {limit}% memory limit")
        env = os.environ.copy()
        env.update({
            "GPU_MAX_HEAP_SIZE": limit,
            "GPU_SINGLE_ALLOC_PERCENT": limit,
            "HIP_LAUNCH_BLOCKING": "1",
        })

        try:
            result = subprocess.run([sys.executable, "test.py"], env=env,
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"  SUCCESS with {limit}% memory limit!")
            else:
                print(f"  Failed with {limit}% memory limit")
                # Check for specific error
                if "out of memory" in result.stderr.lower():
                    print("  -> Out of memory error confirmed")
        except subprocess.TimeoutExpired:
            print(f"  Timeout with {limit}% memory limit")

def run_minimal_gdb():
    """Super minimal GDB - just get crash location"""
    print("\n5. Minimal GDB crash location (5-10x slower max)...")
    print("="*60)

    gdb_cmd = """
echo "Starting program..."
set environment HIP_LAUNCH_BLOCKING 1
set pagination off
run
echo "\n=== CRASH LOCATION ===\n"
where 1
echo "\n=== GPU INFO ===\n"
info cuda kernels
quit
"""

    with open("minimal.gdb", "w") as f:
        f.write(gdb_cmd)

    try:
        result = subprocess.run(
            ["rocgdb", "-batch", "-x", "minimal.gdb", "--args", sys.executable, "test.py"],
            capture_output=True, text=True, timeout=30
        )

        # Extract just the important parts
        for line in result.stdout.split('\n'):
            if any(x in line for x in ['#0', 'received signal', 'kernel:', 'at ']):
                print(line)

    except subprocess.TimeoutExpired:
        print("Timeout after 30 seconds - too slow")
    finally:
        if os.path.exists("minimal.gdb"):
            os.remove("minimal.gdb")

def check_kernel_names():
    """Just list what kernels are being compiled"""
    print("\n6. Check kernel compilation...")
    print("="*60)

    env = os.environ.copy()
    env.update({
        "HIP_TRACE_API": "1",
        "HIP_LAUNCH_BLOCKING": "1",
    })

    result = subprocess.run([sys.executable, "test.py"], env=env,
                          capture_output=True, text=True, timeout=10)

    # Look for kernel launches
    kernels = []
    for line in result.stderr.split('\n'):
        if 'kernel' in line.lower() or 'launch' in line.lower():
            if line not in kernels:
                kernels.append(line)

    print(f"Found {len(kernels)} unique kernel references:")
    for k in kernels[:10]:  # Just first 10
        print(f"  {k[:100]}")

def main():
    print("Lightweight AMD GPU Debugger")
    print("============================\n")

    # Start with fastest checks
    crashed = run_with_basic_check()

    if crashed:
        print("\nCrash confirmed. Running targeted diagnostics...\n")

        # Try these in order of speed
        run_with_hip_check()
        run_with_stack_check()
        run_with_memory_limit()
        check_kernel_names()

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("Run 'python lightweight_debug.py' to repeat")
        print("If stack size changes helped -> stack overflow issue")
        print("If memory limit changes helped -> allocation size issue")
        print("If HIP errors shown -> specific API call issue")

        # Only run slow GDB if user wants
        response = input("\nRun minimal GDB check? (slower but exact location) [y/N]: ")
        if response.lower() == 'y':
            run_minimal_gdb()
    else:
        print("\nNo crash detected with basic settings!")

if __name__ == "__main__":
    main()