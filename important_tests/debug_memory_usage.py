#!/usr/bin/env python
"""
Debug memory usage on AMD GPU for single vs multiple conditions
"""
import os
import subprocess
import sys

def create_test_script(num_conditions):
    """Create a test script with specified number of conditions"""
    if num_conditions == 1:
        conditions_str = "{v.T: 1000, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}"
    else:
        conditions_str = f"{{v.T: np.linspace(900, 1200, {num_conditions}), v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}}"

    script = f'''
import os
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
os.environ['HSA_ENABLE_DEBUG'] = '1'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Try to get memory info before
try:
    import pyhip
    props = pyhip.hipGetDeviceProperties(0)
    print(f"GPU Total Memory: {{props.totalGlobalMem / (1024**3):.2f}} GB")
except:
    pass

db = Database("Al-Cu-Fe.tdb")

print("Testing with {num_conditions} condition(s)...")
result = equilibrium(
    db,
    ['AL', 'CU', 'FE', 'VA'],
    'LIQUID',
    {conditions_str},
    gpu=True,
    verbose=True
)
print(f"Success! GM = {{result.GM.values.flat[0]}}")
'''

    filename = f"test_{num_conditions}_conditions.py"
    with open(filename, "w") as f:
        f.write(script)
    return filename

def run_with_memory_tracking(test_file):
    """Run test with ROCgdb memory tracking"""

    gdb_script = f"""
# Enable memory tracking
set environment HIP_LAUNCH_BLOCKING 1
set environment HSA_ENABLE_DEBUG 1
set environment AMD_LOG_LEVEL 2
set pagination off

# Define function to check memory
define checkmem
  echo \\n=== GPU MEMORY INFO ===\\n
  # Try to get memory info via HIP runtime
  call (void*)hipMemGetInfo(&$free, &$total)
  if $free != 0
    printf "Free memory: %ld MB\\n", $free / 1048576
    printf "Total memory: %ld MB\\n", $total / 1048576
    printf "Used memory: %ld MB\\n", ($total - $free) / 1048576
  end

  # Also check via info cuda
  info cuda malloc
  info cuda devices
end

# Run program
echo \\n=== STARTING PROGRAM ===\\n
run

# If it crashes, get info
echo \\n=== CRASH INFO ===\\n
info registers
bt
checkmem

# Try to get kernel info
info cuda kernels
info cuda blocks
info cuda threads

quit
"""

    with open("memory_debug.gdb", "w") as f:
        f.write(gdb_script)

    cmd = ["rocgdb", "-batch", "-x", "memory_debug.gdb", "--args", sys.executable, test_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract memory info
    print("\n" + "="*60)
    print(f"Results for {test_file}:")
    print("="*60)

    # Look for memory usage
    for line in result.stdout.split('\n'):
        if any(word in line.lower() for word in ['memory', 'free', 'used', 'total', 'malloc', 'fault', 'bytes']):
            print(line)

    # Look for crash location
    crash_found = False
    for line in result.stdout.split('\n'):
        if 'signal' in line.lower() or 'fault' in line.lower():
            crash_found = True
        if crash_found and '#0' in line:
            print(f"CRASH AT: {line}")
            break

    return result.returncode == 0

def check_with_rocm_smi():
    """Use rocm-smi to check memory"""
    print("\nChecking with rocm-smi...")
    try:
        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                              capture_output=True, text=True, timeout=5)
        print(result.stdout)
    except:
        print("rocm-smi not available")

def check_with_hip_trace():
    """Track memory allocations with HIP trace"""
    print("\nTracking allocations with HIP trace...")

    trace_script = '''
import os
os.environ['HIP_TRACE_API'] = '1'
os.environ['HIP_TRACE_API_COLOR'] = 'none'
os.environ['HIP_API_FILTERING'] = 'hipMalloc:hipFree:hipMemcpy'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

db = Database("Al-Cu-Fe.tdb")

# Test 1 condition
print("\\n=== 1 CONDITION ===")
result = equilibrium(db, ['AL','CU','FE','VA'], 'LIQUID',
                    {v.T: 1000, v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}, gpu=True)

# Test 2 conditions
print("\\n=== 2 CONDITIONS ===")
result = equilibrium(db, ['AL','CU','FE','VA'], 'LIQUID',
                    {v.T: [1000, 1100], v.P: 101325, v.X('AL'): 0.3, v.X('CU'): 0.3}, gpu=True)
'''

    with open("trace_allocations.py", "w") as f:
        f.write(trace_script)

    result = subprocess.run([sys.executable, "trace_allocations.py"],
                          capture_output=True, text=True)

    # Count allocations
    allocs_1 = []
    allocs_2 = []
    section = 0

    for line in result.stderr.split('\n'):
        if '1 CONDITION' in line:
            section = 1
        elif '2 CONDITIONS' in line:
            section = 2
        elif 'hipMalloc' in line:
            # Extract size if possible
            import re
            size_match = re.search(r'size=(\d+)', line)
            if size_match:
                size = int(size_match.group(1))
                if section == 1:
                    allocs_1.append(size)
                elif section == 2:
                    allocs_2.append(size)

    print(f"\n1 condition: {len(allocs_1)} allocations, {sum(allocs_1)/(1024**2):.1f} MB total")
    print(f"2 conditions: {len(allocs_2)} allocations, {sum(allocs_2)/(1024**2):.1f} MB total")

    if allocs_2 and allocs_1:
        print(f"Difference: {(sum(allocs_2) - sum(allocs_1))/(1024**2):.1f} MB")

def main():
    print("AMD GPU Memory Usage Debugger")
    print("="*60)

    # Method 1: ROCgdb memory tracking
    print("\n1. Testing with ROCgdb memory tracking...")

    # Test 1 condition
    test1 = create_test_script(1)
    success1 = run_with_memory_tracking(test1)

    # Test 2 conditions
    test2 = create_test_script(2)
    success2 = run_with_memory_tracking(test2)

    print(f"\n1 condition: {'SUCCESS' if success1 else 'FAILED'}")
    print(f"2 conditions: {'SUCCESS' if success2 else 'FAILED'}")

    # Method 2: ROCm-smi
    check_with_rocm_smi()

    # Method 3: HIP trace
    check_with_hip_trace()

    # Clean up
    for f in ['test_1_conditions.py', 'test_2_conditions.py', 'memory_debug.gdb', 'trace_allocations.py']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()