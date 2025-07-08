#!/usr/bin/env python
"""Fix the thread_spec initialization issue after removing sys_spec pointer."""

import os

print("Fixing thread_spec initialization issue...")
print("=" * 80)

gpu_codegen_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"

# Read the file
with open(gpu_codegen_path, 'r') as f:
    content = f.read()

# Find and fix the problematic line
problematic_line = "            SystemSpecification thread_spec = *sys_spec;  // Copy global spec"

if problematic_line in content:
    print("Found problematic thread_spec initialization")
    
    # Replace with code that reads from sys_spec_data
    fix = """            // CRITICAL: Create thread-local SystemSpecification from sys_spec_data
            SystemSpecification thread_spec;
            memset(&thread_spec, 0, sizeof(SystemSpecification));
            
            // Read fields from sys_spec_data array
            int spec_offset = 0;
            thread_spec.num_statevars = (int)sys_spec_data[spec_offset++];
            thread_spec.num_components = (int)sys_spec_data[spec_offset++];
            thread_spec.prescribed_system_amount = sys_spec_data[spec_offset++];
            
            // Read initial_chemical_potentials
            for (int i = 0; i < MAX_COMPONENTS; ++i) {{
                thread_spec.initial_chemical_potentials[i] = sys_spec_data[spec_offset++];
            }}
            
            // Read prescribed_mole_fraction_coefficients matrix
            for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {{
                for (int j = 0; j < MAX_COMPONENTS; ++j) {{
                    thread_spec.prescribed_mole_fraction_coefficients[i][j] = sys_spec_data[spec_offset++];
                }}
            }}
            
            // Read prescribed_mole_fraction_rhs
            for (int i = 0; i < MAX_FIXED_MOLE_FRACTION_CONDITIONS; ++i) {{
                thread_spec.prescribed_mole_fraction_rhs[i] = sys_spec_data[spec_offset++];
            }}
            
            // Read remaining fields
            thread_spec.num_prescribed_mole_fraction_conditions = (int)sys_spec_data[spec_offset++];
            thread_spec.num_prescribed_mole_fraction_coefficients_cols = (int)sys_spec_data[spec_offset++];
            
            // Read index arrays
            for (int i = 0; i < MAX_COMPONENTS; ++i) {{
                thread_spec.free_chemical_potential_indices[i] = (int)sys_spec_data[spec_offset++];
            }}
            thread_spec.num_free_chemical_potentials = (int)sys_spec_data[spec_offset++];
            
            for (int i = 0; i < MAX_STATEVARS; ++i) {{
                thread_spec.free_statevar_indices[i] = (int)sys_spec_data[spec_offset++];
            }}
            thread_spec.num_free_statevars = (int)sys_spec_data[spec_offset++];
            
            for (int i = 0; i < MAX_COMPONENTS; ++i) {{
                thread_spec.fixed_chemical_potential_indices[i] = (int)sys_spec_data[spec_offset++];
            }}
            thread_spec.num_fixed_chemical_potentials = (int)sys_spec_data[spec_offset++];
            
            for (int i = 0; i < MAX_STATEVARS; ++i) {{
                thread_spec.fixed_statevar_indices[i] = (int)sys_spec_data[spec_offset++];
            }}
            thread_spec.num_fixed_statevars = (int)sys_spec_data[spec_offset++];
            
            for (int i = 0; i < MAX_PHASES; ++i) {{
                thread_spec.fixed_stable_compset_indices[i] = (int)sys_spec_data[spec_offset++];
            }}
            thread_spec.num_fixed_stable_compsets = (int)sys_spec_data[spec_offset++];
            thread_spec.max_num_free_stable_phases = (int)sys_spec_data[spec_offset++];
            thread_spec.ALLOWED_MASS_RESIDUAL = sys_spec_data[spec_offset++];"""
    
    content = content.replace(problematic_line, fix)
    print("Replaced with proper field-by-field initialization")
else:
    print("ERROR: Could not find the problematic line!")

# Write the fixed content
with open(gpu_codegen_path, 'w') as f:
    f.write(content)

print("\nFile updated successfully!")
print("The thread_spec initialization issue should now be fixed.")