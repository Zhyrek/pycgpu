#!/usr/bin/env python3
"""Test applying the fix to generated code"""

from pycalphad.gpu.gpu_codegen import fix_hessian_spurious_terms_post_conversion
import re

# Read the generated code
with open('generated_equilibrium_kernel.cu', 'r') as f:
    content = f.read()

# Find out[18] line
for line in content.split('\n'):
    if 'out[18] =' in line:
        original = line.strip()
        break

print("Original out[18]:")
print(f"Length: {len(original)}")
x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', original))
print(f"pow(x[4], (-1)) count: {x4_count}")

# Extract just the expression (after "out[18] = ")
expr = original.split(' = ', 1)[1].rstrip(';')

# Apply the fix
fixed = fix_hessian_spurious_terms_post_conversion(expr, 3, 3, num_statevars=3, debug=True)

print("\nAfter fix:")
x4_count_after = len(re.findall(r'pow\(x\[4\], \(-1\)\)', fixed))
print(f"pow(x[4], (-1)) count: {x4_count_after}")

if x4_count_after < x4_count:
    print(f"\nSUCCESS: Removed {x4_count - x4_count_after} spurious terms!")
    
    # Write the fixed version
    with open('generated_equilibrium_kernel_fixed.cu', 'w') as f:
        f.write(content.replace(original, f"    out[18] = {fixed};"))
    print("Saved fixed version to generated_equilibrium_kernel_fixed.cu")
else:
    print("\nFAILED: No spurious terms removed")