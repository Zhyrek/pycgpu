#!/usr/bin/env python
"""Script to find unwrapped GPU DEBUG statements in gpu_codegen.py"""

import re

def find_unwrapped_debug_statements(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    unwrapped = []
    in_verbose_block = False
    verbose_depth = 0
    ifdef_stack = []  # Track all #ifdef types
    
    for i, line in enumerate(lines):
        # Track all #ifdef blocks
        if '#ifdef' in line:
            ifdef_stack.append(line.strip())
            if 'VERBOSE_DEBUG' in line:
                in_verbose_block = True
                verbose_depth += 1
        elif '#endif' in line and ifdef_stack:
            last_ifdef = ifdef_stack.pop() if ifdef_stack else ""
            if 'VERBOSE_DEBUG' in last_ifdef and verbose_depth > 0:
                verbose_depth -= 1
                if verbose_depth == 0:
                    in_verbose_block = False
        
        # Look for GPU DEBUG printf statements
        if 'printf' in line and 'GPU DEBUG:' in line:
            # Check if it's already in a VERBOSE_DEBUG block
            if not in_verbose_block:
                # Look back up to 10 lines to see if there's a conditional
                has_conditional = False
                for j in range(max(0, i-10), i):
                    if re.search(r'if\s*\(\s*(tid|thread_id|condition_idx|verbose)', lines[j]):
                        has_conditional = True
                        break
                
                # Report the finding
                context_start = max(0, i-2)
                context_end = min(len(lines), i+3)
                context = ''.join(lines[context_start:context_end])
                
                unwrapped.append({
                    'line_num': i + 1,
                    'line': line.strip(),
                    'has_conditional': has_conditional,
                    'context': context,
                    'in_verbose': in_verbose_block,
                    'verbose_depth': verbose_depth
                })
    
    return unwrapped

if __name__ == "__main__":
    filename = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py"
    unwrapped_statements = find_unwrapped_debug_statements(filename)
    
    print(f"Found {len(unwrapped_statements)} potentially unwrapped GPU DEBUG statements:\n")
    
    # Group by whether they have conditionals
    with_conditional = [s for s in unwrapped_statements if s['has_conditional']]
    without_conditional = [s for s in unwrapped_statements if not s['has_conditional']]
    
    print(f"With thread/condition checks: {len(with_conditional)}")
    print(f"Without any conditionals: {len(without_conditional)}")
    
    print("\n=== Statements with conditionals (may be intentionally limited to certain threads) ===")
    for stmt in with_conditional[:5]:  # Show first 5
        print(f"\nLine {stmt['line_num']}: {stmt['line']}")
        
    print(f"\n... and {len(with_conditional) - 5} more")
    
    print("\n=== Statements without any conditionals (should be wrapped) ===")
    for stmt in without_conditional[:10]:  # Show first 10
        print(f"\nLine {stmt['line_num']}: {stmt['line']}")
        print("Context:")
        print(stmt['context'])
        print("-" * 80)
    
    if len(without_conditional) > 10:
        print(f"\n... and {len(without_conditional) - 10} more")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total GPU DEBUG statements: {len(unwrapped_statements)}")
    print(f"Already conditional (tid/thread_id checks): {len(with_conditional)}")
    print(f"Need wrapping with #ifdef VERBOSE_DEBUG: {len(without_conditional)}")