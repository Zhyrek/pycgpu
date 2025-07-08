#!/usr/bin/env python
"""Fix SystemSpecification constructor issue in device code."""

import os

print("Fixing SystemSpecification constructor issue...")
print("=" * 80)

gpu_codegen_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"

# Read the file
with open(gpu_codegen_path, 'r') as f:
    content = f.read()

# Fix 1: In solve_equilibrium_at_condition_global_mem
old_code1 = """    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // Instead, cast to double array and read fields by offset
    const double* spec_data = (const double*)global_spec_base;
    SystemSpecification current_spec;"""

new_code1 = """    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // Instead, cast to double array and read fields by offset
    const double* spec_data = (const double*)global_spec_base;
    SystemSpecification current_spec = {};  // Use C-style initialization to avoid constructor"""

if old_code1 in content:
    content = content.replace(old_code1, new_code1)
    print("Fixed SystemSpecification initialization in solve_equilibrium_at_condition_global_mem")
else:
    print("WARNING: Could not find first initialization to fix")

# Fix 2: In top_level_equilibrium_kernel
old_code2 = """            // CRITICAL: Create thread-local SystemSpecification from sys_spec_data
            SystemSpecification thread_spec;"""

new_code2 = """            // CRITICAL: Create thread-local SystemSpecification from sys_spec_data
            SystemSpecification thread_spec = {};  // Use C-style initialization to avoid constructor"""

if old_code2 in content:
    content = content.replace(old_code2, new_code2)
    print("Fixed SystemSpecification initialization in top_level_equilibrium_kernel")
else:
    print("WARNING: Could not find second initialization to fix")

# Write the fixed content
with open(gpu_codegen_path, 'w') as f:
    f.write(content)

print("\nFile updated successfully!")
print("The SystemSpecification constructor issue should now be fixed.")