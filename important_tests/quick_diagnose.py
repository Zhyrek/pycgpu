#!/usr/bin/env python
"""
Quick AMD GPU diagnostic - no timeouts, minimal overhead
"""

import subprocess
import sys
import os

def run_test(env_vars={}, description=""):
    """Run test with given environment, no timeout"""
    if description:
        print(f"\n{description}")
        print("-" * 40)

    env = os.environ.copy()
    env.update(env_vars)

    # No timeout - let it run or crash
    result = subprocess.run([sys.executable, "test.py"],
                          env=env,
                          capture_output=True,
                          text=True)

    if result.returncode != 0:
        # Just show last 5 lines of error
        stderr_lines = result.stderr.strip().split('\n')
        for line in stderr_lines[-5:]:
            if line.strip():
                print(line)
    else:
        print("SUCCESS!")

    return result.returncode == 0

# Test 1: Just run it normally to confirm it crashes
print("Quick AMD GPU Diagnostic")
print("=" * 60)

if not run_test({}, "1. Normal run (confirm crash):"):
    print("Crash confirmed.\n")

    # Test 2: Single thread only
    if run_test({"HIP_LAUNCH_BLOCKING": "1", "CUDA_LAUNCH_BLOCKING": "1"},
                "2. With blocking launches:"):
        print("→ Works with blocking! Likely a synchronization issue.")

    # Test 3: Check with minimal logging
    run_test({"AMD_LOG_LEVEL": "1"}, "3. Minimal AMD logging:")

else:
    print("No crash detected!")