#!/usr/bin/env python3
"""Check the cleanup in detail"""

import re

# Read the generated code
with open('generated_equilibrium_kernel.cu', 'r') as f:
    code = f.read()

# Find out[18] line
lines = code.split('\n')
for i, line in enumerate(lines):
    if 'out[18]' in line and '=' in line:
        print(f"Found out[18] at line {i+1}")
        
        # Extract the expression
        match = re.match(r'\s*out\[18\]\s*=\s*(.+);', line)
        if match:
            expr = match.group(1)
            
            # Count spurious terms
            x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', expr))
            x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', expr))
            
            print(f"Before cleanup - pow(x[3], (-1)): {x3_count}")
            print(f"Before cleanup - pow(x[4], (-1)): {x4_count}")
            
            # Test the pattern matches
            pattern = r'1\.0\*\(\(1e-15 < x\[4\]\) \? \(pow\(x\[4\], \(-1\)\)\) : 0\)'
            matches = re.findall(pattern, expr)
            print(f"Pattern matches: {len(matches)}")
            
            # Show a snippet of where these terms appear
            idx = expr.find('pow(x[4], (-1))')
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(expr), idx + 50)
                print(f"\nSnippet around first pow(x[4], (-1)):")
                print(f"...{expr[start:end]}...")
                
        break

# Also check out[24] for Y_TI diagonal
for i, line in enumerate(lines):
    if 'out[24]' in line and '=' in line:
        print(f"\nFound out[24] at line {i+1}")
        
        # Extract the expression
        match = re.match(r'\s*out\[24\]\s*=\s*(.+);', line)
        if match:
            expr = match.group(1)
            
            # Count spurious terms
            x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', expr))
            x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', expr))
            
            print(f"Before cleanup - pow(x[3], (-1)): {x3_count}")
            print(f"Before cleanup - pow(x[4], (-1)): {x4_count}")
            
        break