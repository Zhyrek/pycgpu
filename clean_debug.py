#!/usr/bin/env python3
"""Clean all GPU DEBUG statements from code"""

import re
import sys

def clean_incomplete_statements(content):
    """Remove incomplete debug statements left after printf removal"""
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip lines that look like incomplete statements (end with comma or have trailing arguments)
        if re.search(r'^\s+[a-zA-Z_][a-zA-Z0-9_\[\]>-]*(\.[a-zA-Z_][a-zA-Z0-9_\[\]>-]*)*,?\s*$', line):
            # This looks like a trailing argument from a removed printf
            i += 1
            continue
            
        # Skip lines that are just arguments without semicolon
        if re.search(r'^\s+[^;]+\);\s*$', line) and not re.search(r'^\s*(if|for|while|return|break|continue)', line):
            # Check if this is part of an incomplete statement
            if i > 0 and re.search(r'[,\(]\s*$', lines[i-1].rstrip()):
                i += 1
                continue
                
        result.append(line)
        i += 1
    
    return '\n'.join(result)

def process_file(filename):
    """Process a file to clean up debug statements"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except:
        print(f"Could not read {filename}")
        return
        
    # Clean up incomplete statements
    cleaned = clean_incomplete_statements(content)
    
    with open(filename, 'w') as f:
        f.write(cleaned)
    
    print(f"Cleaned {filename}")

if __name__ == '__main__':
    files = [
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py',
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_equilibrium.py',
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h',
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/eqsolver.h'
    ]
    
    for f in files:
        process_file(f)