#!/usr/bin/env python
"""Disable debug output in CPU code."""

import re

# Files to clean up
files_to_clean = [
    'pycalphad/core/minimizer.pyx',
    'pycalphad/core/composition_set.pyx',
    'pycalphad/core/solver.pyx'
]

for filename in files_to_clean:
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Comment out print statements that start with [CPU
        content = re.sub(r'^(\s*)(print\(.*?\[CPU.*?\))', r'\1# \2', content, flags=re.MULTILINE)
        
        # Also comment out CPU HESSIAN DEBUG, CPU ADVANCE, etc
        content = re.sub(r'^(\s*)(if.*debug.*:)', r'\1# \2', content, flags=re.MULTILINE)
        
        with open(filename, 'w') as f:
            f.write(content)
            
        print(f"Cleaned {filename}")
    except Exception as e:
        print(f"Error cleaning {filename}: {e}")