#!/usr/bin/env python3
"""Verify the fix worked"""

import re

# Read generated code
with open('generated_equilibrium_kernel.cu', 'r') as f:
    lines = f.readlines()

# Find out[18] and out[24]
out18 = None
out24 = None

for line in lines:
    if 'out[18] =' in line:
        out18 = line.strip()
    elif 'out[24] =' in line:
        out24 = line.strip()

print("=== Checking out[18] (Y_NB diagonal) ===")
if out18:
    x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', out18))
    x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', out18))
    print(f"pow(x[3], (-1)) count: {x3_count} (expected: some)")
    print(f"pow(x[4], (-1)) count: {x4_count} (expected: 0)")
    
print("\n=== Checking out[24] (Y_TI diagonal) ===")
if out24:
    x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', out24))
    x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', out24))
    print(f"pow(x[3], (-1)) count: {x3_count} (expected: 0)")
    print(f"pow(x[4], (-1)) count: {x4_count} (expected: some)")

# Also check if the fix messages appear in stderr
print("\n=== Checking for fix messages ===")
import subprocess
result = subprocess.run(['python', 'save_generated_cuda.py'], capture_output=True, text=True)
if 'Removed' in result.stderr:
    print("Found removal messages in output:")
    for line in result.stderr.split('\n'):
        if 'Removed' in line:
            print(f"  {line}")
else:
    print("No removal messages found")