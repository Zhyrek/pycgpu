#!/usr/bin/env python3
"""Test that the final cleanup is removing spurious terms"""

from pycalphad.gpu.gpu_codegen import _final_hessian_cleanup
import re

# Read the generated code
with open('generated_equilibrium_kernel.cu', 'r') as f:
    code = f.read()

print("=== BEFORE CLEANUP ===")
# Count spurious terms in out[18] and out[24]
for out_idx in [18, 24]:
    pattern = rf'out\[{out_idx}\]\s*='
    for line in code.split('\n'):
        if re.search(pattern, line):
            x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', line))
            x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', line))
            print(f"out[{out_idx}]: pow(x[3], (-1)) = {x3_count}, pow(x[4], (-1)) = {x4_count}")
            break

# Apply cleanup
cleaned_code = _final_hessian_cleanup(code)

print("\n=== AFTER CLEANUP ===")
# Count spurious terms in cleaned code
for out_idx in [18, 24]:
    pattern = rf'out\[{out_idx}\]\s*='
    for line in cleaned_code.split('\n'):
        if re.search(pattern, line):
            x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', line))
            x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', line))
            print(f"out[{out_idx}]: pow(x[3], (-1)) = {x3_count}, pow(x[4], (-1)) = {x4_count}")
            break

# Save cleaned code for inspection
with open('generated_equilibrium_kernel_cleaned.cu', 'w') as f:
    f.write(cleaned_code)