#!/usr/bin/env python
"""Test the fix_missing_operators function"""

import re

def fix_missing_operators(code_str: str) -> str:
    """Fix missing operators between numbers in generated code.
    
    This handles cases where dependent variable substitution or other
    transformations leave bare numbers next to each other without operators.
    """
    import re
    
    # Fix patterns like "9.0* 1 11.0*" -> "9.0* 1 + 11.0*"
    code_str = re.sub(r'(\* )\s*(\d+)\s+(\d+\.?\d*\*)', r'\1\2 + \3', code_str)
    
    # Fix patterns like ") + 1 11.0*" -> ") + 1 + 11.0*"
    code_str = re.sub(r'(\) \+ )\s*(\d+)\s+(\d+\.?\d*\*)', r'\1\2 + \3', code_str)
    
    # Fix patterns like "+ 1 11.0*" -> "+ 1 + 11.0*"
    code_str = re.sub(r'(\+ )\s*(\d+)\s+(\d+\.?\d*\*)', r'\1\2 + \3', code_str)
    
    # Fix patterns where a number is followed by another number with * like "1 11.0*"
    # but be careful not to match things like "x[1]" or decimal numbers
    code_str = re.sub(r'(?<=[^0-9.\[\]])(\d+)\s+(\d+\.?\d*\*)', r'\1 + \2', code_str)
    
    return code_str

# Test cases
test_cases = [
    "9.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1 11.0*((1e-15 < x[5]) ? (pow(x[5], (-1))) : 0))",
    "74.8305*x[2]*(9.0* 1 11.0*((1e-15 < x[5]) ? (pow(x[5], (-1))) : 0))",
    "x[2]*(9.0* 1 11.0*((1e-15 < x[5])",
]

print("Testing fix_missing_operators:")
print("="*60)

for i, test in enumerate(test_cases):
    print(f"\nTest {i+1}:")
    print(f"Input:  {test}")
    result = fix_missing_operators(test)
    print(f"Output: {result}")
    if "1 11.0*" in result or "1 11.0*" in result:
        print("FAILED - Pattern still present!")
    else:
        print("OK")