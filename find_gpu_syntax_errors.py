#!/usr/bin/env python
"""Find specific syntax errors in GPU generated code that cause nvcc compilation failures."""

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
print("Finding Syntax Errors in Generated GPU Code")
print("="*80)

# Test with all phases
wks = Workspace(db, components, phases, conditions)

print(f"\nGenerating code for {len(phases)} phases...")

try:
    # Generate code
    result = _generate_c_code_for_phase_models(wks, include_hess=True, validate=False)
    all_device_functions, init_calls, unique_models, phase_map = result
    
    print(f"Generated code successfully")
    print(f"Total code length: {len(all_device_functions)} characters")
    
    # Common error patterns that nvcc would complain about
    error_patterns = [
        # Pattern 1: Adjacent numbers without operators (after fix_missing_operators should be rare)
        (r'(\d+\.?\d*)\s+(\d+\.?\d*)(?![eE\-\+])', 'Adjacent numbers without operator'),
        
        # Pattern 2: Number directly followed by identifier (no space/operator)
        (r'(\d+\.?\d*)([a-zA-Z_]\w*)', 'Number directly before identifier'),
        
        # Pattern 3: Missing semicolons (lines that should end with ; but don't)
        (r'^[^{}/\*#]\s*\w.*[^;{}\s]\s*$', 'Possible missing semicolon'),
        
        # Pattern 4: Unclosed parentheses/brackets
        (r'\([^)]*$|^[^(]*\)', 'Unbalanced parentheses'),
        
        # Pattern 5: Invalid operators (e.g., ++x[i] which might be generated incorrectly)
        (r'\+\+x\[|--x\[', 'Invalid increment/decrement on array element'),
        
        # Pattern 6: Empty function calls
        (r'\w+\(\s*\)', 'Empty function call (might need void)'),
        
        # Pattern 7: Double operators that aren't valid
        (r'[^\+\-\*/=!<>]\*\*(?!\*)', 'Invalid ** operator (use pow)'),
        
        # Pattern 8: Missing return in non-void functions
        (r'__device__\s+double\s+\w+[^{]+\{[^}]*\}', 'Check for missing return'),
    ]
    
    # Split code into lines for analysis
    lines = all_device_functions.split('\n')
    
    print(f"\nAnalyzing {len(lines)} lines of code...")
    
    all_errors = []
    
    # Check each pattern
    for pattern, description in error_patterns:
        print(f"\nChecking for: {description}")
        errors = []
        
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('//'):
                continue
            
            try:
                if description == 'Adjacent numbers without operator':
                    # Special handling for this pattern
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        # Make sure it's not scientific notation
                        if not re.match(r'\d+\.?\d*[eE][\+\-]?\d+', match.group(0)):
                            errors.append((i+1, line.strip(), match.group(0)))
                
                elif description == 'Number directly before identifier':
                    # Special handling to exclude valid cases
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        num = match.group(1)
                        ident = match.group(2)
                        # Exclude scientific notation, type suffixes, and valid syntax
                        if (ident[0] not in 'eEfFlLuU' and 
                            not re.search(r'[*+\-/,;(){}=\s]', line[match.start(2)-1:match.start(2)])):
                            errors.append((i+1, line.strip(), f"{num}{ident}"))
                
                elif description == 'Possible missing semicolon':
                    # Check if line should end with semicolon
                    stripped = line.strip()
                    if (stripped and 
                        not stripped.endswith((';', '{', '}', ':', ',')) and
                        not stripped.startswith(('if', 'else', 'for', 'while', '#', '//', 'case', 'default')) and
                        'return' in stripped or '=' in stripped or '(' in stripped):
                        errors.append((i+1, line.strip(), "missing ;"))
                
                elif re.search(pattern, line):
                    errors.append((i+1, line.strip(), re.search(pattern, line).group(0)))
            
            except Exception as e:
                # Skip lines that cause regex errors
                pass
        
        if errors:
            print(f"  Found {len(errors)} issues:")
            for line_num, line_text, match in errors[:5]:
                print(f"    Line {line_num}: {match}")
                if len(line_text) > 80:
                    print(f"      {line_text[:80]}...")
                else:
                    print(f"      {line_text}")
            if len(errors) > 5:
                print(f"    ... and {len(errors)-5} more")
            all_errors.extend([(description, e) for e in errors])
    
    # Look for very long expressions that might cause issues
    print("\n" + "="*80)
    print("CHECKING FOR EXTREMELY LONG EXPRESSIONS")
    print("="*80)
    
    # Find expressions by looking for return statements or assignments
    long_expressions = []
    current_expr = []
    in_expr = False
    
    for i, line in enumerate(lines):
        if 'return' in line or '=' in line:
            in_expr = True
            current_expr = [line]
        elif in_expr:
            if ';' in line:
                current_expr.append(line)
                expr_text = '\n'.join(current_expr)
                if len(expr_text) > 5000:  # Very long expression
                    long_expressions.append((i+1, len(expr_text), expr_text[:200]))
                in_expr = False
                current_expr = []
            else:
                current_expr.append(line)
    
    if long_expressions:
        print(f"\nFound {len(long_expressions)} very long expressions:")
        for line_num, length, preview in long_expressions[:5]:
            print(f"  Line {line_num}: {length} characters")
            print(f"    Preview: {preview}...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_errors:
        error_counts = {}
        for desc, _ in all_errors:
            error_counts[desc] = error_counts.get(desc, 0) + 1
        
        print("\nError counts by type:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
        
        print(f"\nTotal errors found: {len(all_errors)}")
        print("\nThese errors are likely causing the nvcc compilation failures.")
    else:
        print("\nNo obvious syntax errors found.")
        print("The nvcc errors might be due to:")
        print("  1. Extremely long expressions exceeding compiler limits")
        print("  2. Deeply nested conditional expressions")
        print("  3. Other CUDA-specific limitations")
    
    # Save a sample of problematic code
    if all_errors:
        with open("gpu_syntax_errors.txt", "w") as f:
            f.write("GPU Code Syntax Errors\n")
            f.write("="*80 + "\n\n")
            
            for i, (desc, (line_num, line_text, match)) in enumerate(all_errors[:50]):
                f.write(f"Error {i+1}: {desc}\n")
                f.write(f"Line {line_num}: {match}\n")
                f.write(f"Context: {line_text}\n")
                f.write("-"*60 + "\n")
        
        print(f"\nSaved first 50 errors to: gpu_syntax_errors.txt")
        
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()