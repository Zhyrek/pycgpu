#!/usr/bin/env python3
"""Better fix for GPU hessian spurious terms"""

def fix_hessian_spurious_terms_v2(hess_str, i_idx, j_idx):
    """
    Remove spurious entropy cross-terms from diagonal hessian elements.
    
    For d²G/dY_i², remove RT/Y_j terms where j != i.
    """
    import re
    
    # Only fix diagonal elements for site fractions (indices >= 3)
    if i_idx != j_idx or i_idx < 3:
        return hess_str
    
    # The most direct approach: for hessian[i,i], remove pow(x[j], (-1)) where j != i
    # This handles the spurious RT/Y_j terms
    
    # Find all indices used in pow(x[idx], (-1)) patterns
    inverse_pattern = r'pow\(x\[(\d+)\], \(-1\)\)'
    
    def should_remove_term(match):
        """Check if this inverse term should be removed"""
        idx = int(match.group(1))
        # Remove if it's a different site fraction
        return idx >= 3 and idx != i_idx
    
    # First, let's identify terms that contain spurious inverses
    # Simpler approach: just look for the pattern and extract the index
    
    # Find all terms like: 1.0*((1e-15 < x[N]) ? (pow(x[N], (-1))) : 0)
    # where N is a digit
    terms_to_remove = []
    
    # Use a simpler pattern
    import re
    # This matches the whole term including coefficient
    simple_pattern = r'[\d.]+\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\1\], \(-1\)\)\) : 0\)'
    
    for match in re.finditer(simple_pattern, hess_str):
        idx = int(match.group(1))
        if idx >= 3 and idx != i_idx:
            # This is a spurious term
            terms_to_remove.append(match.group(0))
            print(f"DEBUG: Found spurious term for x[{idx}] in hessian[{i_idx},{j_idx}]")
    
    # Remove the spurious terms
    fixed = hess_str
    for term in terms_to_remove:
        # Remove the term and any preceding ' + '
        fixed = fixed.replace(' + ' + term, '')
        fixed = fixed.replace(term + ' + ', '')
        fixed = fixed.replace(term, '')
    
    # Clean up any double spaces or operators
    fixed = re.sub(r'\s+', ' ', fixed)
    fixed = re.sub(r'\(\s*\+', '(', fixed)
    fixed = re.sub(r'\+\s*\)', ')', fixed)
    
    # Special case: if we removed all terms from a sum, we might have empty parentheses
    fixed = re.sub(r'8\.3145\*x\[2\]\*\(\s*\)/\(x\[3\] \+ x\[4\]\)', '0', fixed)
    
    # CRITICAL: Also remove the /(x[3] + x[4]) divisor from entropy terms
    # This divisor should have been canceled by the (x[3] + x[4]) factor at the front
    # Debug: print what we're trying to match
    print(f"DEBUG: Before divisor removal: {fixed[:100]}...")
    
    # Simpler approach: just replace the specific divisor
    if '/(x[3] + x[4])' in fixed:
        fixed = fixed.replace('/(x[3] + x[4])', '')
        print("DEBUG: Removed /(x[3] + x[4]) divisor")
    
    return fixed


# Test the fix
if __name__ == "__main__":
    # Test case 1: hessian[3,3] should only have x[3] inverse
    test1 = '8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'
    fixed1 = fix_hessian_spurious_terms_v2(test1, 3, 3)
    print("Test 1 - hessian[3,3]:")
    print("Original:", test1)
    print("Fixed:   ", fixed1)
    print("Contains x[4]^-1?", 'pow(x[4], (-1))' in fixed1)
    print("Contains /(x[3] + x[4])?", '/(x[3] + x[4])' in fixed1)
    print()
    
    # Test case 2: hessian[4,4] should only have x[4] inverse
    test2 = '8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'
    fixed2 = fix_hessian_spurious_terms_v2(test2, 4, 4)
    print("Test 2 - hessian[4,4]:")
    print("Original:", test2)
    print("Fixed:   ", fixed2)
    print("Contains x[3]^-1?", 'pow(x[3], (-1))' in fixed2)
    print("Contains /(x[3] + x[4])?", '/(x[3] + x[4])' in fixed2)
    
    # Verify the fix removes the correct terms
    assert 'pow(x[4], (-1))' not in fixed1, "Test 1 failed: still contains x[4] inverse"
    assert 'pow(x[3], (-1))' not in fixed2, "Test 2 failed: still contains x[3] inverse"
    assert '/(x[3] + x[4])' not in fixed1, "Test 1 failed: still contains divisor"
    assert '/(x[3] + x[4])' not in fixed2, "Test 2 failed: still contains divisor"
    print("\nAll tests passed!")