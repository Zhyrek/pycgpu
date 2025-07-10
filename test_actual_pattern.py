#!/usr/bin/env python3
"""Test the actual pattern from out[18]"""

import re

# Read the actual expression
with open('out18_expression.txt', 'r') as f:
    test_str = f.read().strip()

# Remove the "out[18] = " prefix
test_str = test_str.split(' = ', 1)[1]

print(f"Expression length: {len(test_str)} chars")

# The pattern from the fix
inv_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : 0\)'

# Find matches
matches = list(re.finditer(inv_pattern, test_str))
print(f"\nFound {len(matches)} matches with original pattern")

if matches:
    for i, match in enumerate(matches[:5]):
        idx = match.group(1)
        print(f"  Match {i}: index={idx}")

# Double-check by looking for the exact string
exact_str = '1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0)'
count = test_str.count(exact_str)
print(f"\nExact string count for x[4]: {count}")

# Look for the pattern without escaping
simple_count = test_str.count('pow(x[4], (-1))')
print(f"Simple pow(x[4], (-1)) count: {simple_count}")

# Check what's around one of these terms
idx = test_str.find('pow(x[4], (-1))')
if idx >= 0:
    start = max(0, idx - 60)
    end = min(len(test_str), idx + 40)
    print(f"\nContext around first pow(x[4], (-1)):")
    print(f"'{test_str[start:end]}'")