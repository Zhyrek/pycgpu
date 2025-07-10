#!/usr/bin/env python3
"""Test pattern matching for variable names"""

import re

# Test expression with variable names
test_expr = (
    "8.3145*T*(1.0*((1e-15 < BCC_A20TI) ? (pow(BCC_A20TI, (-1))) : 0) + "
    "1.0*((1e-15 < BCC_A20NB) ? (pow(BCC_A20NB, (-1))) : 0))/(BCC_A20NB + BCC_A20TI)"
)

print("Test expression:")
print(test_expr)
print()

# Test pattern for BCC_A20TI
var_pattern = r'BCC_A2\d*TI'
spurious_pattern = rf'1\.0\*\(\(1e-15 < ({var_pattern})\) \? \(pow\(\1, \(-1\)\)\) : 0\)'

print(f"Pattern: {spurious_pattern}")
matches = list(re.finditer(spurious_pattern, test_expr))
print(f"Found {len(matches)} matches")

if matches:
    for match in matches:
        print(f"  Match: {match.group(0)}")
        print(f"  Variable: {match.group(1)}")

# Try simpler pattern
print("\nTrying simpler pattern...")
simple_pattern = r'1\.0\*\(\(1e-15 < (BCC_A2\d*TI)\) \? \(pow\((BCC_A2\d*TI), \(-1\)\)\) : 0\)'
matches2 = list(re.finditer(simple_pattern, test_expr))
print(f"Found {len(matches2)} matches")

# Even simpler - just find the terms
print("\nJust finding BCC_A20TI terms...")
ti_pattern = r'BCC_A20TI'
ti_matches = re.findall(ti_pattern, test_expr)
print(f"Found {len(ti_matches)} BCC_A20TI occurrences")