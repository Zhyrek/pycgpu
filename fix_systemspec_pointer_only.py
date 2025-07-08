#!/usr/bin/env python
"""Fix SystemSpecification usage to only use pointers, avoiding constructor calls."""

import os

print("Fixing SystemSpecification to use pointers only...")
print("=" * 80)

gpu_codegen_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"

# Read the file
with open(gpu_codegen_path, 'r') as f:
    content = f.read()

# Fix 1: In solve_equilibrium_at_condition_global_mem, use byte array on stack
old_code1 = """    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // Instead, cast to double array and read fields by offset
    const double* spec_data = (const double*)global_spec_base;
    SystemSpecification current_spec;
    memset(&current_spec, 0, sizeof(SystemSpecification));  // Initialize to zero"""

new_code1 = """    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // Use a byte array to avoid constructor calls
    const double* spec_data = (const double*)global_spec_base;
    char current_spec_bytes[sizeof(SystemSpecification)];
    memset(current_spec_bytes, 0, sizeof(SystemSpecification));
    SystemSpecification* current_spec_ptr = (SystemSpecification*)current_spec_bytes;
    SystemSpecification& current_spec = *current_spec_ptr;"""

if old_code1 in content:
    content = content.replace(old_code1, new_code1)
    print("Fixed SystemSpecification in solve_equilibrium_at_condition_global_mem")
else:
    print("WARNING: Could not find first code block to fix")

# Fix 2: Similar fix for top_level_equilibrium_kernel
old_code2 = """            // CRITICAL: Create thread-local SystemSpecification from sys_spec_data
            SystemSpecification thread_spec;
            memset(&thread_spec, 0, sizeof(SystemSpecification));  // Initialize to zero"""

new_code2 = """            // CRITICAL: Create thread-local SystemSpecification from sys_spec_data
            char thread_spec_bytes[sizeof(SystemSpecification)];
            memset(thread_spec_bytes, 0, sizeof(SystemSpecification));
            SystemSpecification* thread_spec_ptr = (SystemSpecification*)thread_spec_bytes;
            SystemSpecification& thread_spec = *thread_spec_ptr;"""

if old_code2 in content:
    content = content.replace(old_code2, new_code2)
    print("Fixed SystemSpecification in top_level_equilibrium_kernel")
else:
    print("WARNING: Could not find second code block to fix")

# Write the fixed content
with open(gpu_codegen_path, 'w') as f:
    f.write(content)

print("\nFile updated successfully!")
print("SystemSpecification should now be allocated via byte array to avoid constructor calls.")