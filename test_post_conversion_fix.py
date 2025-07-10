#!/usr/bin/env python3
"""Test fixing spurious terms after variable conversion"""

import re

# Sample expression from generated code (out[18] = hess[3,3])
test_expr = """8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])"""

print("Original expression:")
print(test_expr)
print()

def fix_hessian_spurious_terms_post_conversion(hess_str, i_idx, j_idx, num_statevars=3):
    """
    Remove spurious entropy cross-terms from diagonal hessian elements AFTER variable conversion.
    
    For d²G/dx[i]², remove RT/x[j] terms where j != i.
    This works on the converted x[i] format rather than variable names.
    """
    # Only process diagonal elements
    if i_idx != j_idx:
        return hess_str
        
    # Only process site fraction indices (after state variables)
    if i_idx < num_statevars:
        return hess_str
    
    print(f"[GPU HESSIAN FIX POST] Processing diagonal element x[{i_idx}]")
    
    # For diagonal element x[i], we need to remove 1/x[j] terms where j != i
    # These appear in entropy expressions like:
    # 8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])
    
    # Find all indices that appear in 1/x[j] terms
    inv_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : 0\)'
    
    modified = False
    fixed = hess_str
    
    # Find all such terms
    for match in re.finditer(inv_pattern, hess_str):
        idx_str = match.group(1)
        idx = int(idx_str)
        
        # If this is a different index than our diagonal, it's spurious
        if idx != i_idx and idx >= num_statevars:
            print(f"  Found spurious 1/x[{idx}] term")
            # Remove this term
            term_to_remove = match.group(0)
            
            # Check if it's part of a sum
            # Look for patterns like "term1 + term2" where we need to remove term2
            # or "term2 + term1" where we need to remove term2
            
            # Find the position in the string
            start = match.start()
            end = match.end()
            
            # Check what's before and after
            before = fixed[:start].rstrip()
            after = fixed[end:].lstrip()
            
            if before.endswith('+'):
                # Remove the preceding ' + '
                while before and before[-1] in ' +':
                    before = before[:-1]
                fixed = before + after
                modified = True
            elif after.startswith('+'):
                # Remove the following ' + '
                while after and after[0] in ' +':
                    after = after[1:]
                fixed = before + after
                modified = True
            else:
                # Just remove the term
                fixed = before + ' ' + after
                modified = True
                
            # Update for next iteration
            hess_str = fixed
    
    if modified:
        # Clean up any double spaces or arithmetic issues
        fixed = re.sub(r'\s+', ' ', fixed)
        fixed = re.sub(r'\(\s*\+', '(', fixed)
        fixed = re.sub(r'\+\s*\)', ')', fixed)
        fixed = re.sub(r'\+\s*\+', '+', fixed)
        
        # Special case: if we removed all terms from a sum, we might have just 0
        # e.g., "8.3145*x[2]*(0)/(x[3] + x[4])" -> "0"
        fixed = re.sub(r'[\d.]+\*[^*]+\*\(0\)/[^)]+', '0', fixed)
        
        print(f"  Fixed expression (removed spurious terms)")
    
    return fixed

# Test on Y_NB diagonal (index 3)
fixed = fix_hessian_spurious_terms_post_conversion(test_expr, 3, 3)
print("\nFixed expression:")
print(fixed)

# Verify the x[4] term is gone
if 'pow(x[4], (-1))' not in fixed:
    print("\nSUCCESS: Spurious x[4] term removed!")
else:
    print("\nFAILED: x[4] term still present")

# Also test a more complex expression
print("\n\n=== Testing complex expression ===")
complex_expr = """16.629*x[2]*(1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0) + 1.0*((1e-15 < x[3]) ? (1 + log(x[3])) : 0))/pow((x[3] + x[4]), 2) + 8.3145*x[2]*(1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0) + 1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0))/(x[3] + x[4])"""

print("Original:")
print(complex_expr[:100] + "...")

fixed2 = fix_hessian_spurious_terms_post_conversion(complex_expr, 3, 3)
print("\nFixed:")
print(fixed2[:100] + "...")