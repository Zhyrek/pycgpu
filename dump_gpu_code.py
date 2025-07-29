#!/usr/bin/env python
"""Dump generated GPU C code to files for inspection."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import re

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

# Create workspace with just a few phases
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

# Test with first 5 phases
test_phases = phases[:5]
wks = Workspace(db, components, test_phases, conditions)

print(f"Generating GPU code for phases: {test_phases}")
print("="*80)

# Generate code
try:
    phase_codes = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    print(f"Generated {len(phase_codes)} functions")
    
    # Group functions by phase
    functions_by_phase = {}
    for func_name, func_code in phase_codes.items():
        # Extract phase name from function name
        for phase in test_phases:
            if phase in func_name:
                if phase not in functions_by_phase:
                    functions_by_phase[phase] = {}
                functions_by_phase[phase][func_name] = func_code
                break
    
    # Examine each phase
    for phase_idx, (phase_name, phase_functions) in enumerate(functions_by_phase.items()):
        print(f"\n{'='*80}")
        print(f"PHASE {phase_idx}: {phase_name}")
        print(f"Functions: {list(phase_functions.keys())}")
        print(f"{'='*80}")
        
        # Look at the energy function (_obj)
        obj_func = None
        for func_name, func_code in phase_functions.items():
            if '_obj' in func_name and '_formulaobj' not in func_name:
                obj_func = (func_name, func_code)
                break
        
        if obj_func:
            func_name, func_code = obj_func
            print(f"\nEnergy function: {func_name}")
            print("-"*60)
            
            lines = func_code.split('\n')
            
            # Show first 40 lines
            print("First 40 lines:")
            for i, line in enumerate(lines[:40]):
                print(f"{i+1:3d}: {line}")
            
            # Check for systematic issues
            print("\n" + "-"*60)
            print("Checking for systematic issues:")
            
            issues = []
            
            # 1. Number followed by identifier without operator
            for i, line in enumerate(lines):
                # Look for patterns like "123.45identifier" or "123 identifier"
                matches = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z_]\w*)', line)
                for match in matches:
                    # Check if it's not a valid pattern like "123.45*var" or function call
                    full_match = match[0] + match[1]
                    if not re.search(r'\d+\.?\d*\s*[\*\+\-\/]\s*[a-zA-Z_]', line) and \
                       not re.search(r'\d+\.?\d*[eE][\+\-]?\d+', full_match) and \
                       match[1] not in ['e', 'E', 'f', 'F', 'l', 'L', 'u', 'U']:
                        issues.append(f"Line {i+1}: Possible missing operator: '{match[0]} {match[1]}'")
            
            # 2. Multiple operators in sequence
            for i, line in enumerate(lines):
                if re.search(r'[\+\-\*]{2,}', line) and not re.search(r'/\*|\*/', line):
                    issues.append(f"Line {i+1}: Multiple operators: {line.strip()[:50]}")
            
            # 3. Unbalanced parentheses per line
            for i, line in enumerate(lines):
                open_count = line.count('(')
                close_count = line.count(')')
                if open_count != close_count and '//' not in line:
                    issues.append(f"Line {i+1}: Unbalanced parens ({open_count} open, {close_count} close)")
            
            # Print first 10 issues
            if issues:
                print("\nIssues found:")
                for issue in issues[:10]:
                    print(f"  {issue}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more issues")
            else:
                print("\nNo obvious issues found")
        
        # Save to file for detailed inspection
        with open(f"gpu_code_{phase_name}.c", "w") as f:
            f.write(f"// Generated GPU code for phase {phase_name}\n\n")
            for func_name, func_code in phase_functions.items():
                f.write(f"// Function: {func_name}\n")
                f.write(func_code)
                f.write("\n\n")
        print(f"\nSaved to: gpu_code_{phase_name}.c")
    
except Exception as e:
    print(f"Error generating code: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()