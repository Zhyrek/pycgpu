#!/usr/bin/env python
"""Simple check for GPU code generation issues."""

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

print("Checking GPU code generation...")

# Test with first few phases to avoid broken pipe
test_phases = phases[:5]
wks = Workspace(db, components, test_phases, conditions, verbose=False)

try:
    # Generate code
    result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    all_device_functions, init_calls, unique_models, phase_map = result
    
    print(f"\nGenerated code length: {len(all_device_functions)} characters")
    
    # Check for common syntax errors
    lines = all_device_functions.split('\n')
    
    # Pattern 1: Numbers directly before identifiers
    pattern1_errors = []
    for i, line in enumerate(lines):
        matches = re.finditer(r'(\d+\.?\d*)([a-zA-Z_]\w*)', line)
        for match in matches:
            num = match.group(1)
            ident = match.group(2)
            # Check if it's not scientific notation or valid syntax
            if (ident[0] not in 'eEfFlLuU' and 
                match.start(2) > 0 and
                line[match.start(2)-1] not in '*+-/,;(){}= \t'):
                pattern1_errors.append((i+1, f"{num}{ident}", line.strip()[:80]))
    
    if pattern1_errors:
        print(f"\nFound {len(pattern1_errors)} cases of numbers directly before identifiers:")
        for line_num, match, context in pattern1_errors[:10]:
            print(f"  Line {line_num}: '{match}' in: {context}...")
    
    # Pattern 2: Check for extremely long lines (nvcc might have limits)
    long_lines = [(i+1, len(line)) for i, line in enumerate(lines) if len(line) > 1000]
    if long_lines:
        print(f"\nFound {len(long_lines)} extremely long lines:")
        for line_num, length in long_lines[:5]:
            print(f"  Line {line_num}: {length} characters")
    
    # Pattern 3: Check for deeply nested expressions
    max_nesting = 0
    for line in lines:
        nesting = 0
        max_line_nesting = 0
        for char in line:
            if char == '(':
                nesting += 1
                max_line_nesting = max(max_line_nesting, nesting)
            elif char == ')':
                nesting -= 1
        max_nesting = max(max_nesting, max_line_nesting)
    
    print(f"\nMaximum parenthesis nesting depth: {max_nesting}")
    if max_nesting > 50:
        print("  WARNING: Very deep nesting might cause nvcc issues")
    
    # Save a sample function for inspection
    func_start = all_device_functions.find('__device__')
    if func_start >= 0:
        func_end = all_device_functions.find('}\n\n', func_start)
        if func_end > func_start:
            sample_func = all_device_functions[func_start:func_end+1]
            with open("sample_gpu_function.c", "w") as f:
                f.write("// Sample GPU generated function\n")
                f.write(sample_func)
            print("\nSaved sample function to: sample_gpu_function.c")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {str(e)}")
    
print("\nDone.")