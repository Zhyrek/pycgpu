#!/usr/bin/env python3
"""Final removal of all GPU DEBUG statements."""

import re

def remove_all_gpu_debug(content):
    """Remove all GPU DEBUG printf statements and their enclosing blocks."""
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for standalone printf with GPU DEBUG
        if 'printf' in line and 'GPU DEBUG' in line:
            # Skip to the end of this printf statement
            while i < len(lines) and not lines[i].rstrip().endswith(';'):
                i += 1
            i += 1  # Skip the line with semicolon
            continue
            
        # Check for if blocks
        if_match = re.match(r'^(\s*)if\s*\([^)]+\)\s*\{\s*$', line)
        if if_match:
            indent = if_match.group(1)
            # Collect the block
            block_start = i
            i += 1
            brace_count = 1
            has_gpu_debug = False
            has_other_content = False
            
            while i < len(lines) and brace_count > 0:
                current = lines[i]
                if 'GPU DEBUG' in current:
                    has_gpu_debug = True
                elif current.strip() and not current.strip().startswith('//'):
                    # Check if it's not just printf or braces
                    if not re.match(r'^\s*(printf|for\s*\(|}\s*$|{\s*$)', current):
                        has_other_content = True
                        
                brace_count += current.count('{') - current.count('}')
                i += 1
            
            # If block only has debug, skip it
            if has_gpu_debug and not has_other_content:
                result.append(indent + "// Debug output removed")
                continue
            else:
                # Keep the block
                result.extend(lines[block_start:i])
                continue
        
        # Keep regular lines
        result.append(line)
        i += 1
    
    return '\n'.join(result)

def process_file(filename):
    """Process a file to remove debug statements."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Repeatedly process until no more changes
    prev_content = ""
    while content != prev_content:
        prev_content = content
        content = remove_all_gpu_debug(content)
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Processed {filename}")

if __name__ == '__main__':
    files = [
        '/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py',
    ]
    
    for f in files:
        process_file(f)