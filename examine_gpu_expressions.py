#!/usr/bin/env python
"""Examine the generated C expressions for each phase to find systematic errors."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import re

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Create workspace
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("="*80)
print("Examining Generated C Expressions for Al-Cu-Fe Phases")
print("="*80)

# Test with first few phases
test_phases = phases[:10]
wks = Workspace(db, components, test_phases, conditions)

print(f"\nGenerating code for phases: {test_phases}")

try:
    # Generate code - returns a tuple
    result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    all_device_functions, init_calls, unique_models, phase_map = result
    
    print(f"\nGenerated code successfully")
    print(f"Device functions code length: {len(all_device_functions)}")
    print(f"Number of init calls: {len(init_calls)}")
    print(f"Number of unique models: {len(unique_models)}")
    
    # Parse the device functions to extract individual functions
    print("\n" + "="*80)
    print("EXTRACTING AND ANALYZING INDIVIDUAL FUNCTIONS")
    print("="*80)
    
    # Split by __device__ to find individual functions
    function_blocks = all_device_functions.split('__device__')
    
    print(f"\nFound {len(function_blocks)-1} device functions")
    
    # Analyze each function
    for i, func_block in enumerate(function_blocks[1:]):  # Skip first empty split
        # Extract function name
        lines = func_block.strip().split('\n')
        if not lines:
            continue
            
        # First line should have the function signature
        func_signature = lines[0]
        
        # Try to extract function name
        match = re.search(r'(\w+)\s*\(', func_signature)
        if match:
            func_name = match.group(1)
        else:
            func_name = f"unknown_function_{i}"
        
        print(f"\n{'='*60}")
        print(f"Function {i}: {func_name}")
        print(f"{'='*60}")
        
        # Reconstruct full function
        full_func = "__device__ " + func_block
        
        # Show first 20 lines
        func_lines = full_func.split('\n')
        print("First 20 lines:")
        for j, line in enumerate(func_lines[:20]):
            print(f"{j+1:3d}: {line}")
        
        # Look for systematic issues
        print("\nChecking for issues:")
        issues = []
        
        # 1. Adjacent numbers without operators
        for j, line in enumerate(func_lines):
            # Skip comments and empty lines
            if line.strip().startswith('//') or not line.strip():
                continue
            
            # Look for patterns like "123.45 678.9" or "123 identifier"
            matches = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*|\w+)', line)
            for match in matches:
                # Check if second part is a number or starts with letter
                if match[1][0].isdigit() or match[1][0].isalpha():
                    # Make sure it's not scientific notation or valid syntax
                    context = line[max(0, line.find(match[0])-5):line.find(match[0])+len(match[0])+len(match[1])+5]
                    if not re.search(r'[*+\-/,;(){}]', context[5:6]):  # No operator between
                        issues.append(f"Line {j+1}: Missing operator between '{match[0]}' and '{match[1]}'")
        
        # 2. Show expression complexity
        if '_obj' in func_name and 'formula' not in func_name:
            # This is likely the main energy function
            # Count operations
            plus_count = full_func.count('+')
            mult_count = full_func.count('*')
            pow_count = full_func.count('pow(')
            log_count = full_func.count('log(')
            
            print(f"\nExpression statistics:")
            print(f"  - Additions: {plus_count}")
            print(f"  - Multiplications: {mult_count}")
            print(f"  - Power operations: {pow_count}")
            print(f"  - Log operations: {log_count}")
            print(f"  - Total characters: {len(full_func)}")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} potential issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues)-5} more")
        
        # Save problematic functions
        if issues or len(full_func) > 10000:
            filename = f"gpu_func_{func_name}.c"
            with open(filename, 'w') as f:
                f.write(f"// Function: {func_name}\n")
                f.write(f"// Issues found: {len(issues)}\n")
                f.write(f"// Length: {len(full_func)} characters\n\n")
                f.write(full_func)
            print(f"\nSaved to: {filename}")
    
    # Also check the init calls
    print("\n" + "="*80)
    print("PHASE RECORD INITIALIZATION CALLS")
    print("="*80)
    
    for i, init_call in enumerate(init_calls[:10]):
        print(f"\n{i}: {init_call.strip()}")
        
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Check the generated .c files for functions with issues.")
print("Common problems to look for:")
print("  1. Missing operators between numbers")
print("  2. Extremely long expressions (>10000 chars)")
print("  3. Syntax errors in generated code")