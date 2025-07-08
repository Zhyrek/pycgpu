#!/usr/bin/env python3
"""Remove ALL debug printf statements from GPU code."""

import re

def remove_debug_printfs(filepath):
    """Remove debug printf statements from a file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip lines with printf
        if 'printf' in line and not 'fprintf' in line:  # Keep fprintf for error reporting
            # Check if it's inside an if (thread_id == 0) block
            # Look backwards for the if statement
            found_thread_check = False
            for j in range(max(0, i-5), i):
                if 'if (thread_id == 0)' in lines[j] or 'if (tid == 0' in lines[j]:
                    found_thread_check = True
                    break
            
            if found_thread_check:
                # Skip the printf line
                i += 1
                continue
            elif 'thread_id == 0' not in line and 'tid == 0' not in line:
                # Also skip standalone printf statements
                i += 1
                continue
        
        # Check for empty if blocks that might have been left behind
        if re.match(r'^\s*if\s*\([^)]*thread_id == 0[^)]*\)\s*\{\s*$', line):
            # Check if the next non-empty line is a closing brace
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and lines[j].strip() == '}':
                # Skip the empty if block
                i = j + 1
                continue
        
        result.append(line)
        i += 1
    
    # Clean up the result
    content = '\n'.join(result)
    
    # Remove any remaining empty if (thread_id == 0) blocks
    content = re.sub(r'if\s*\([^)]*thread_id == 0[^)]*\)\s*\{\s*\}', '', content)
    content = re.sub(r'if\s*\([^)]*tid == 0[^)]*\)\s*\{\s*\}', '', content)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return filepath

def main():
    files_to_clean = [
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/eqsolver.h',
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h',
    ]
    
    for filepath in files_to_clean:
        print(f"Cleaning {filepath}...")
        remove_debug_printfs(filepath)
    
    print("\nDone! Checking for remaining printf statements:")
    
    for filepath in files_to_clean:
        with open(filepath, 'r') as f:
            content = f.read()
        printf_count = len(re.findall(r'\bprintf\s*\(', content))
        print(f"{filepath}: {printf_count} printf statements remaining")

if __name__ == '__main__':
    main()