#!/usr/bin/env python
"""Debug GPU code generation by intercepting the generated C code."""

from pycalphad import Database
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import _generate_c_code_for_phase_models
import pycalphad.variables as v
import re

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

print("="*80)
print("Debugging GPU Code Generation for Al-Cu-Fe")
print("="*80)

# Create workspace
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

wks = Workspace(db, components, phases, conditions)

# Generate code for all phases
print("\nGenerating C code for all phases...")
try:
    phase_function_defs = _generate_c_code_for_phase_models(wks, include_hess=True, validate=True)
    print(f"Successfully generated {len(phase_function_defs)} function definitions")
except Exception as e:
    print(f"ERROR during code generation: {type(e).__name__}: {str(e)}")
    phase_function_defs = {}

# Analyze the generated code
print("\n" + "="*80)
print("ANALYZING GENERATED CODE:")
print("="*80)

# Extract individual functions
all_functions = []
current_func = []
current_name = None

for line in phase_function_defs.values():
    if isinstance(line, str):
        for subline in line.split('\n'):
            if '__device__' in subline and '(' in subline:
                # New function starts
                if current_func and current_name:
                    all_functions.append((current_name, '\n'.join(current_func)))
                # Extract function name
                match = re.search(r'(\w+)\s*\(', subline)
                current_name = match.group(1) if match else 'unknown'
                current_func = [subline]
            elif current_func:
                current_func.append(subline)

# Add last function
if current_func and current_name:
    all_functions.append((current_name, '\n'.join(current_func)))

print(f"\nFound {len(all_functions)} functions")

# Check each function for common issues
issues_by_type = {
    'adjacent_numbers': [],
    'double_operators': [],
    'unbalanced_parens': [],
    'invalid_vars': [],
    'syntax_errors': []
}

for func_name, func_code in all_functions[:10]:  # Check first 10 functions
    print(f"\n{'='*60}")
    print(f"Function: {func_name}")
    print(f"{'='*60}")
    
    # Show first 300 chars
    print("Code preview:")
    print(func_code[:300] + "..." if len(func_code) > 300 else func_code)
    
    # Check for issues
    issues = []
    
    # 1. Adjacent numbers without operators (our previous fix target)
    adjacent_nums = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)', func_code)
    if adjacent_nums:
        issues.append(f"Adjacent numbers: {adjacent_nums[0]}")
        issues_by_type['adjacent_numbers'].append((func_name, adjacent_nums[0]))
    
    # 2. Double operators
    double_ops = re.findall(r'[\+\-\*]{2,}(?![/=])', func_code)
    if double_ops:
        issues.append(f"Double operators: {double_ops[0]}")
        issues_by_type['double_operators'].append((func_name, double_ops[0]))
    
    # 3. Unbalanced parentheses
    open_p = func_code.count('(')
    close_p = func_code.count(')')
    if open_p != close_p:
        issues.append(f"Parens: {open_p} open, {close_p} close")
        issues_by_type['unbalanced_parens'].append((func_name, f"{open_p} vs {close_p}"))
    
    # 4. Variables starting with digits
    invalid_vars = re.findall(r'\b\d+[a-zA-Z_]\w*\b', func_code)
    if invalid_vars:
        issues.append(f"Invalid vars: {invalid_vars[0]}")
        issues_by_type['invalid_vars'].append((func_name, invalid_vars[0]))
    
    # 5. Common C syntax errors
    # Missing semicolons (check for lines without ; that should have them)
    lines = func_code.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.endswith((';', '{', '}', '*/')) and \
           not stripped.startswith(('if', 'else', 'for', 'while', '//', '#')):
            # Check if next line starts with an operator (continuation)
            if i + 1 < len(lines) and not lines[i+1].strip().startswith(('+', '-', '*', '/', ')', ',')):
                issues.append(f"Missing semicolon: line {i}")
                issues_by_type['syntax_errors'].append((func_name, f"line {i}: {stripped[:50]}"))
                break
    
    if issues:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

# Summary
print("\n" + "="*80)
print("SUMMARY OF SYSTEMATIC ISSUES:")
print("="*80)

for issue_type, occurrences in issues_by_type.items():
    if occurrences:
        print(f"\n{issue_type.upper()} ({len(occurrences)} occurrences):")
        for func, example in occurrences[:3]:  # Show first 3
            print(f"  - {func}: {example}")
        if len(occurrences) > 3:
            print(f"  ... and {len(occurrences) - 3} more")

# Try to find the actual compilation error by looking at the code more carefully
print("\n" + "="*80)
print("LOOKING FOR SPECIFIC SYNTAX PATTERNS:")
print("="*80)

# Check for specific patterns that might cause nvcc errors
patterns_to_check = [
    (r'(?<!\w)(\d+)\s*([a-zA-Z_]\w*)', 'Number directly before identifier'),
    (r'(\+\s*-|\-\s*\+)', 'Adjacent plus/minus operators'),
    (r'(\*\s*\*)', 'Double multiplication'),
    (r'([^=!<>])=([^=])', 'Single = in expression context'),
    (r'\)\s*\w+\s*\(', 'Missing operator between ) and identifier('),
]

for pattern, description in patterns_to_check:
    print(f"\n{description}:")
    found = False
    for func_name, func_code in all_functions[:20]:
        matches = re.findall(pattern, func_code)
        if matches:
            print(f"  - {func_name}: {matches[0]}")
            found = True
            break
    if not found:
        print("  - None found")