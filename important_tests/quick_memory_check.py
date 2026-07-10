#!/usr/bin/env python
"""
Quick memory usage check without debugger
"""
import os
import subprocess
import sys

# Set environment for memory tracking
env = os.environ.copy()
env.update({
    "HIP_LAUNCH_BLOCKING": "1",
    "ROCR_VISIBLE_DEVICES": "0",
    "HSA_TOOLS_LIB": "libroctracer64.so",
    "ROCTRACER_DOMAIN": "hip",
    "HSA_TOOLS_REPORT_LOAD_FAILURE": "1",
    "AMD_LOG_LEVEL": "3",
    # Track allocations
    "HIP_TRACE_API": "1",
    "HIP_TRACE_ACTIVITY": "1",
    "HIP_PRINT_ENV": "1",  # Print HIP configuration
})

test_script = '''
import sys

def get_gpu_memory():
    """Try to get GPU memory info"""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        free, total = device.mem_info
        print(f"[MEMORY] Free: {free/(1024**2):.1f} MB, Total: {total/(1024**2):.1f} MB, Used by CuPy: {mempool.used_bytes()/(1024**2):.1f} MB")
        return free, total
    except Exception as e:
        print(f"[MEMORY] Could not get memory info: {e}")
        return 0, 0

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import cupy as cp

db = Database("Al-Cu-Fe.tdb")

print("\\n=== TESTING 1 CONDITION ===")
get_gpu_memory()

try:
    result = equilibrium(db, ['AL','CU','FE','VA'], 'LIQUID',
                        {v.T: 1000, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}, gpu=True)
    print(f"SUCCESS - GM = {result.GM.values.flat[0]}")
except Exception as e:
    print(f"FAILED: {e}")

get_gpu_memory()

print("\\n=== TESTING 2 CONDITIONS ===")
get_gpu_memory()

try:
    result = equilibrium(db, ['AL','CU','FE','VA'], 'LIQUID',
                        {v.T: [1000, 1100], v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}, gpu=True)
    print(f"SUCCESS - GM values: {result.GM.values.flatten()}")
except Exception as e:
    print(f"FAILED: {e}")

get_gpu_memory()

# Force memory pool stats
try:
    mempool = cp.get_default_memory_pool()
    print(f"\\n[MEMORY POOL STATS]")
    print(f"  Total allocated: {mempool.total_bytes()/(1024**2):.1f} MB")
    print(f"  Currently used: {mempool.used_bytes()/(1024**2):.1f} MB")
except:
    pass
'''

with open("memory_test.py", "w") as f:
    f.write(test_script)

# Run and capture all output
print("Running memory test...")
print("="*60)

result = subprocess.run([sys.executable, "memory_test.py"], env=env,
                       capture_output=True, text=True)

# Parse output for memory allocations from HIP trace
print("\nMemory Allocations (from HIP trace):")
allocations = []
for line in result.stderr.split('\n'):
    if 'hipMalloc' in line or 'cuMemAlloc' in line:
        # Try to extract size
        import re
        # Look for patterns like size=12345 or (12345)
        size_match = re.search(r'size[=:]\s*(\d+)|bytes[=:]\s*(\d+)|\((\d+)\)', line)
        if size_match:
            size = int(next(g for g in size_match.groups() if g))
            allocations.append(size)
            if size > 1024*1024:  # Only show allocations > 1MB
                print(f"  {size/(1024**2):.1f} MB allocation")

if allocations:
    print(f"\nTotal allocated: {sum(allocations)/(1024**2):.1f} MB across {len(allocations)} allocations")
    print(f"Largest allocation: {max(allocations)/(1024**2):.1f} MB")

# Show program output
print("\nProgram output:")
print("-"*60)
print(result.stdout)

# Show any errors
if result.stderr:
    errors = [l for l in result.stderr.split('\n')
              if 'error' in l.lower() or 'fault' in l.lower() or 'failed' in l.lower()]
    if errors:
        print("\nErrors detected:")
        for e in errors[:10]:  # First 10 errors
            print(f"  {e}")

# Clean up
if os.path.exists("memory_test.py"):
    os.remove("memory_test.py")