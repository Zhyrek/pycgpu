#!/usr/bin/env python
"""Fix for GPU memory access issue - change how SystemSpecification is passed to GPU."""

import os
import shutil
from datetime import datetime

print("Fixing GPU memory access issue...")
print("=" * 80)

# Backup the file first
gpu_codegen_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"
backup_path = gpu_codegen_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(gpu_codegen_path, backup_path)
print(f"Created backup: {backup_path}")

# Read the file
with open(gpu_codegen_path, 'r') as f:
    content = f.read()

# Find the problematic line
problematic_line = "    SystemSpecification current_spec = *global_spec_base;"
if problematic_line in content:
    print("\nFound problematic dereference at line:")
    print(f"  {problematic_line}")
    
    # Replace with safer approach that reads fields individually
    safer_code = """    // CRITICAL FIX: Safely read SystemSpecification from GPU memory
    // Cannot dereference struct pointer directly due to alignment/memory access issues
    // Instead, cast to double array and read fields by offset
    const double* spec_data = (const double*)global_spec_base;
    SystemSpecification current_spec;
    
    // Initialize to safe defaults first
    memset(&current_spec, 0, sizeof(SystemSpecification));
    
    // Read scalar fields from known offsets (based on struct layout)
    // These offsets must match the Python packing in _create_system_specification_struct
    int offset = 0;
    current_spec.num_statevars = (int)spec_data[offset++];
    current_spec.num_components = (int)spec_data[offset++];
    current_spec.prescribed_system_amount = spec_data[offset++];
    
    // Read initial_chemical_potentials array
    for (int i = 0; i < MAX_COMPONENTS; ++i) {{
        current_spec.initial_chemical_potentials[i] = spec_data[offset++];
    }}
    
    // Read prescribed_mole_fraction_coefficients matrix
    for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {{
        for (int j = 0; j < MAX_COMPONENTS; ++j) {{
            current_spec.prescribed_mole_fraction_coefficients[i][j] = spec_data[offset++];
        }}
    }}
    
    // Read prescribed_mole_fraction_rhs array
    for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {{
        current_spec.prescribed_mole_fraction_rhs[i] = spec_data[offset++];
    }}
    
    // Read remaining scalar fields
    current_spec.num_prescribed_mole_fraction_conditions = (int)spec_data[offset++];
    current_spec.num_prescribed_mole_fraction_coefficients_cols = (int)spec_data[offset++];
    
    // Read index arrays
    for (int i = 0; i < MAX_COMPONENTS; ++i) {{
        current_spec.free_chemical_potential_indices[i] = (int)spec_data[offset++];
    }}
    current_spec.num_free_chemical_potentials = (int)spec_data[offset++];
    
    for (int i = 0; i < MAX_STATEVARS; ++i) {{
        current_spec.free_statevar_indices[i] = (int)spec_data[offset++];
    }}
    current_spec.num_free_statevars = (int)spec_data[offset++];
    
    for (int i = 0; i < MAX_COMPONENTS; ++i) {{
        current_spec.fixed_chemical_potential_indices[i] = (int)spec_data[offset++];
    }}
    current_spec.num_fixed_chemical_potentials = (int)spec_data[offset++];
    
    for (int i = 0; i < MAX_STATEVARS; ++i) {{
        current_spec.fixed_statevar_indices[i] = (int)spec_data[offset++];
    }}
    current_spec.num_fixed_statevars = (int)spec_data[offset++];
    
    for (int i = 0; i < MAX_PHASES; ++i) {{
        current_spec.fixed_stable_compset_indices[i] = (int)spec_data[offset++];
    }}
    current_spec.num_fixed_stable_compsets = (int)spec_data[offset++];
    current_spec.max_num_free_stable_phases = (int)spec_data[offset++];
    current_spec.ALLOWED_MASS_RESIDUAL = spec_data[offset++];
    
    // Skip the large SVD arrays - they're allocated separately in global memory"""
    
    content = content.replace(problematic_line, safer_code)
    print("\nReplaced with safer field-by-field reading approach")
else:
    print("\nERROR: Could not find the problematic line!")
    print("The code may have already been modified.")

# Also fix the similar issue in the top-level kernel where sys_spec is dereferenced
top_level_issue = """        const SystemSpecification* sys_spec = (const SystemSpecification*)global_spec_ptr_raw;
        int num_statevars = sys_spec->num_statevars;"""

if top_level_issue in content:
    print("\nFound similar issue in top-level kernel")
    safer_top_level = """        // CRITICAL FIX: Safely read num_statevars from spec data
        const double* sys_spec_data = (const double*)global_spec_ptr_raw;
        int num_statevars = (int)sys_spec_data[0];  // First field in SystemSpecification"""
    
    content = content.replace(top_level_issue, safer_top_level)
    print("Fixed top-level kernel dereference")

# Fix another similar dereference
another_issue = "            const SystemSpecification* sys_spec = (const SystemSpecification*)global_spec_ptr_raw;"
if another_issue in content:
    print("\nFound another sys_spec dereference")
    safer_version = """            // CRITICAL FIX: Safely read SystemSpecification fields
            const double* sys_spec_data = (const double*)global_spec_ptr_raw;
            // Read fields by offset: num_statevars=0, num_components=1, prescribed_system_amount=2"""
    content = content.replace(another_issue, safer_version)

# Fix all remaining sys_spec-> dereferences
import re

# Find all sys_spec->field accesses
sys_spec_accesses = re.findall(r'sys_spec->(\w+)', content)
if sys_spec_accesses:
    print(f"\nFound {len(set(sys_spec_accesses))} unique sys_spec field accesses to fix")
    
    # Define field offsets based on struct layout
    field_offsets = {
        'num_statevars': 0,
        'num_components': 1,
        'prescribed_system_amount': 2,
        'num_prescribed_mole_fraction_conditions': 3 + 32 + 32*32 + 32,  # After arrays
        'num_free_chemical_potentials': 3 + 32 + 32*32 + 32 + 1 + 1 + 32 + 1,
        # Add more as needed
    }
    
    # Replace each access
    for field in set(sys_spec_accesses):
        if field in field_offsets:
            offset = field_offsets[field]
            # For scalar fields, cast to int if needed
            if field.startswith('num_'):
                replacement = f"((int)sys_spec_data[{offset}])"
            else:
                replacement = f"sys_spec_data[{offset}]"
            
            pattern = f"sys_spec->{field}"
            content = content.replace(pattern, replacement)
            print(f"  Replaced sys_spec->{field} with {replacement}")

# Write the fixed content
with open(gpu_codegen_path, 'w') as f:
    f.write(content)

print("\nFile updated successfully!")
print("The GPU memory access issue should now be fixed.")
print("\nKey changes:")
print("1. Replaced direct struct pointer dereference with field-by-field reading")
print("2. Uses double array access with known offsets")
print("3. Avoids alignment and memory access issues")
print("\nPlease test the GPU code again.")