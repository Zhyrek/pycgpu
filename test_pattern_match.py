#!/usr/bin/env python3
"""Test if the pattern matches"""

import re

# Get a sample line from the generated code
with open('generated_equilibrium_kernel.cu', 'r') as f:
    lines = f.readlines()
    
# Find out[18] line
for line in lines:
    if 'out[18] =' in line:
        test_str = line.strip()
        break

print("Testing pattern match on out[18] line")
print(f"Line length: {len(test_str)} chars")

# The pattern used in the fix
inv_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : 0\)'

# Find matches
matches = list(re.finditer(inv_pattern, test_str))
print(f"\nFound {len(matches)} matches")

if matches:
    for i, match in enumerate(matches[:5]):
        idx = match.group(1)
        print(f"  Match {i}: index={idx}, text='{match.group(0)[:50]}...'")
else:
    # Try a simpler pattern
    simple_pattern = r'pow\(x\[4\], \(-1\)\)'
    simple_matches = re.findall(simple_pattern, test_str)
    print(f"\nSimpler pattern found {len(simple_matches)} matches")
    
    # Check what the actual pattern looks like
    print("\nSearching for actual pattern...")
    # Extract a portion around pow(x[4]
    idx = test_str.find('pow(x[4], (-1))')
    if idx >= 0:
        start = max(0, idx - 50)
        end = min(len(test_str), idx + 50)
        print(f"Context around pow(x[4], (-1)): '{test_str[start:end]}'")