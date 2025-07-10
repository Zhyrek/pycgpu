#!/usr/bin/env python3
"""Test regex pattern"""

import re

# Test string from generated code
test_str = '8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'

print("Test string:")
print(test_str)
print()

# Pattern from the fix function
inv_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : 0\)'

print(f"Pattern: {inv_pattern}")

# Find matches
matches = list(re.finditer(inv_pattern, test_str))
print(f"\nFound {len(matches)} matches")

if not matches:
    # Try without the \d+ in pow
    alt_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[(\d+)\], \(-1\)\)\) : 0\)'
    alt_matches = list(re.finditer(alt_pattern, test_str))
    print(f"\nAlternative pattern found {len(alt_matches)} matches")
    
    if alt_matches:
        for match in alt_matches:
            print(f"  Index in condition: {match.group(1)}")
            print(f"  Index in pow: {match.group(2)}")
            print(f"  Match text: {match.group(0)}")

# Check if backref is the issue
print("\nTrying with explicit index match...")
simple_pattern = r'1\.0\*\(\(1e-15 < x\[4\]\) \? \(pow\(x\[4\], \(-1\)\)\) : 0\)'
simple_matches = re.findall(simple_pattern, test_str)
print(f"Simple pattern (x[4] only) found {len(simple_matches)} matches")