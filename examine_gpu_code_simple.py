#!/usr/bin/env python
"""Simple examination of GPU generated code to find systematic errors."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import re

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Test with just LIQUID phase first
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU code generation for individual phases...")
print("="*80)

# Test each phase individually
test_phases = ['LIQUID', 'FCC_A1', 'BCC_A2', 'ALCU_ZETA', 'AL2FE']

for phase_name in test_phases:
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"{'='*60}")
    
    try:
        # Create workspace with just this phase
        wks = Workspace(db, components, [phase_name], conditions)
        
        # Generate code
        phase_codes = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
        
        # The result is a dict with function names as keys
        print(f"Generated {len(phase_codes)} functions")
        
        # Find the energy function
        energy_func = None
        for func_name, func_code in phase_codes.items():
            if f'{phase_name}_obj' in func_name and 'formula' not in func_name:
                energy_func = func_code
                print(f"Found energy function: {func_name}")
                break
        
        if energy_func:
            lines = energy_func.split('\n')
            print(f"Function has {len(lines)} lines")
            
            # Show first 30 lines
            print("\nFirst 30 lines:")
            for i, line in enumerate(lines[:30]):
                print(f"{i+1:3d}: {line}")
            
            # Look for systematic issues
            print("\nChecking for issues:")
            
            # Pattern 1: Number directly followed by identifier (missing operator)
            issues = []
            for i, line in enumerate(lines):
                # Skip comments and preprocessor directives
                if line.strip().startswith('//') or line.strip().startswith('#'):
                    continue
                    
                # Look for number followed by letter without operator
                # But exclude scientific notation (e.g., 1.23e-4) and type suffixes (e.g., 123L)
                matches = re.finditer(r'(\d+\.?\d*)([a-zA-Z_]\w*)', line)
                for match in matches:
                    num = match.group(1)
                    ident = match.group(2)
                    # Check if it's not scientific notation or type suffix
                    if ident[0] not in 'eEfFlLuU' and not re.search(r'[*+\-/\s]', line[match.start(2)-1:match.start(2)]):
                        issues.append(f"Line {i+1}: Missing operator between '{num}' and '{ident}'")
            
            if issues:
                print(f"\nFound {len(issues)} potential issues:")
                for issue in issues[:5]:
                    print(f"  {issue}")
            else:
                print("\nNo obvious missing operator issues found")
                
            # Save the function for manual inspection
            filename = f"gpu_{phase_name}_energy.c"
            with open(filename, 'w') as f:
                f.write(f"// Energy function for {phase_name}\n")
                f.write(energy_func)
            print(f"\nSaved to: {filename}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("Summary:")
print("Check the generated .c files for systematic syntax errors")
print("Look especially for:")
print("  1. Numbers directly followed by identifiers (e.g., '123identifier')")
print("  2. Missing operators between terms")
print("  3. Malformed expressions")