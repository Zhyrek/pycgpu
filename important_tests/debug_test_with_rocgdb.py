#!/usr/bin/env python
"""
ROCgdb wrapper for debugging test.py
This script automates running test.py under ROCgdb to catch GPU crashes
"""

import subprocess
import sys
import os
import tempfile

def create_gdb_script():
    """Create a GDB script for automated debugging"""
    gdb_commands = """
# Set environment variables for debugging
set environment HIP_LAUNCH_BLOCKING 1
set environment AMD_LOG_LEVEL 4
set environment HSA_ENABLE_DEBUG 1
set environment AMD_SERIALIZE_KERNEL 3
set environment AMD_SERIALIZE_COPY 3
set environment HIP_ABORT_ON_ERROR 1

# Set breakpoints on common error conditions
catch throw
catch signal SIGSEGV
catch signal SIGABRT
catch signal SIGBUS

# Configure CUDA/HIP debugging
set cuda break_on_launch kernel_name
set cuda memcheck on
set cuda api_failures stop

# Run the program
run

# When it crashes, automatically collect information
echo \\n=== CRASH DETECTED ===\\n
echo \\n=== BACKTRACE ===\\n
bt full

echo \\n=== CURRENT THREAD INFO ===\\n
info threads
thread apply all bt

echo \\n=== GPU KERNEL INFO ===\\n
info cuda kernels
info cuda threads
info cuda blocks
info cuda devices

echo \\n=== REGISTERS ===\\n
info registers

echo \\n=== MEMORY MAP ===\\n
info proc mappings

echo \\n=== GPU WARPS/WAVES ===\\n
info cuda warps

echo \\n=== LOCAL VARIABLES ===\\n
info locals

echo \\n=== DISASSEMBLY ===\\n
disas $pc-32,$pc+32

# Continue to see if there are more errors
continue
"""
    return gdb_commands

def run_with_rocgdb_interactive():
    """Run test.py with interactive ROCgdb"""
    print("Starting interactive ROCgdb session...")
    print("="*60)
    print("Commands to use once in ROCgdb:")
    print("  run           - Start the program")
    print("  bt            - Show backtrace when crashed")
    print("  info cuda kernels - Show GPU kernel info")
    print("  info cuda threads - Show GPU thread info")
    print("  quit          - Exit debugger")
    print("="*60)

    cmd = ["rocgdb", "python", "test.py"]
    subprocess.run(cmd)

def run_with_rocgdb_automated():
    """Run test.py with automated ROCgdb crash detection"""
    print("Running automated ROCgdb crash detection...")
    print("="*60)

    # Create temporary GDB script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gdb', delete=False) as f:
        f.write(create_gdb_script())
        gdb_script_path = f.name

    try:
        # Run ROCgdb with the script
        cmd = [
            "rocgdb",
            "-batch",
            "-x", gdb_script_path,
            "--args", sys.executable, "test.py"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print output
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)

        # Save to file for analysis
        with open("rocgdb_crash_report.txt", "w") as f:
            f.write("=== ROCgdb Crash Report ===\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        print("\nCrash report saved to: rocgdb_crash_report.txt")

    finally:
        # Clean up temporary file
        if os.path.exists(gdb_script_path):
            os.remove(gdb_script_path)

def run_with_hip_trace():
    """Run with HIP tracing to see which kernel crashes"""
    print("Running with HIP API tracing...")
    print("="*60)

    env = os.environ.copy()
    env.update({
        "HIP_TRACE_API": "1",
        "HIP_TRACE_ACTIVITY": "1",
        "HIP_LAUNCH_BLOCKING": "1",
        "AMD_LOG_LEVEL": "4",
        "HIP_ABORT_ON_ERROR": "1",
        "AMD_SERIALIZE_KERNEL": "3"
    })

    cmd = [sys.executable, "test.py"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # The last HIP API call before crash is usually the culprit
    print("Last 50 lines before crash:")
    print("="*60)
    lines = result.stderr.split('\n')
    for line in lines[-50:]:
        print(line)

    with open("hip_trace_output.txt", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)

    print("\nFull trace saved to: hip_trace_output.txt")

def run_with_rocprof():
    """Use rocprof to profile and identify crashing kernel"""
    print("Running with ROCprof...")
    print("="*60)

    # Create rocprof input file
    with open("rocprof_input.txt", "w") as f:
        f.write("# Collect basic metrics\n")
        f.write("pmc: SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_VMEM_WR,SQ_INSTS_VMEM_RD\n")
        f.write("kernel: equilibrium_kernel\n")  # Adjust kernel name if known

    cmd = [
        "rocprof",
        "-i", "rocprof_input.txt",
        "-o", "rocprof_output.csv",
        sys.executable, "test.py"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("ROCprof output:")
    print(result.stdout)
    print(result.stderr)

    # Check if output file was created
    if os.path.exists("rocprof_output.csv"):
        print("\nProfiling data saved to: rocprof_output.csv")
        with open("rocprof_output.csv", "r") as f:
            print("First few lines of profiling data:")
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
    else:
        print("\nNo profiling output generated (crash occurred before kernel completion)")

def main():
    print("AMD GPU Debugging Tool for test.py")
    print("===================================\n")

    print("Select debugging method:")
    print("1. Interactive ROCgdb (step through manually)")
    print("2. Automated ROCgdb (collect crash info automatically)")
    print("3. HIP API Trace (see which HIP call crashes)")
    print("4. ROCprof (profile kernel execution)")
    print("5. Run all methods")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        run_with_rocgdb_interactive()
    elif choice == "2":
        run_with_rocgdb_automated()
    elif choice == "3":
        run_with_hip_trace()
    elif choice == "4":
        run_with_rocprof()
    elif choice == "5":
        print("\n--- Running HIP Trace ---")
        run_with_hip_trace()
        print("\n--- Running Automated ROCgdb ---")
        run_with_rocgdb_automated()
        print("\n--- Running ROCprof ---")
        run_with_rocprof()
    else:
        print("Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()