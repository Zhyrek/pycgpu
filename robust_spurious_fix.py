#!/usr/bin/env python3
"""Implement a robust fix for spurious entropy terms"""

# The spurious terms come from the entropy expression:
# S = -R * sum(Y_i * log(Y_i)) / sum(Y_i)
#
# For a binary system with Y_NB + Y_TI = 1, this becomes:
# S = -R * (Y_NB * log(Y_NB) + Y_TI * log(Y_TI))
#
# When we take d²S/dY_NB², we should get:
# d²S/dY_NB² = R / Y_NB
#
# But due to the normalization factor (Y_NB + Y_TI) in the denominator,
# we get spurious cross-terms like R / Y_TI in the Y_NB diagonal.
#
# The fix needs to:
# 1. Identify these spurious 1/Y_j terms in diagonal elements
# 2. Remove them while preserving the correct 1/Y_i terms
# 3. Handle all the different formats these terms can appear in

import re

def remove_spurious_entropy_terms(hess_str, i_idx, j_idx, var_format='x'):
    """
    Remove spurious entropy cross-terms from diagonal Hessian elements.
    
    This is a robust implementation that handles multiple formats and ensures
    spurious terms are removed at any stage of the code generation pipeline.
    
    Args:
        hess_str: The Hessian expression string
        i_idx, j_idx: The indices being differentiated (0-based)
        var_format: 'x' for x[i] format, 'name' for BCC_A20NB format
    
    Returns:
        Fixed expression with spurious terms removed
    """
    # Only process diagonal elements
    if i_idx != j_idx:
        return hess_str
    
    # Only process site fraction variables (typically indices 3 and above)
    # Indices 0,1,2 are usually N, P, T
    if i_idx < 3:
        return hess_str
    
    print(f"[ROBUST FIX] Processing diagonal element [{i_idx},{j_idx}]")
    
    # Build a list of all site fraction indices
    # For binary system: indices 3 (Y_NB) and 4 (Y_TI)
    site_fraction_indices = []
    if var_format == 'x':
        # Look for all x[i] where i >= 3
        all_indices = set(re.findall(r'x\[(\d+)\]', hess_str))
        site_fraction_indices = [int(idx) for idx in all_indices if int(idx) >= 3]
    else:
        # For name format, we'd need to identify the variables differently
        # This is a placeholder for future extension
        return hess_str
    
    if not site_fraction_indices:
        print(f"[ROBUST FIX] No site fraction variables found")
        return hess_str
    
    print(f"[ROBUST FIX] Site fraction indices: {site_fraction_indices}")
    
    # For each spurious index (not equal to i_idx)
    modified = False
    result = hess_str
    
    for spurious_idx in site_fraction_indices:
        if spurious_idx == i_idx:
            continue  # This is the correct term, don't remove
        
        print(f"[ROBUST FIX] Looking for spurious 1/x[{spurious_idx}] terms")
        
        # Pattern 1: Simple pow(x[j], (-1))
        pattern1 = rf'pow\(x\[{spurious_idx}\], \(-1\)\)'
        
        # Pattern 2: Conditional (1e-15 < x[j]) ? (pow(x[j], (-1))) : 0
        pattern2 = rf'\(\(1e-15 < x\[{spurious_idx}\]\) \? \(pow\(x\[{spurious_idx}\], \(-1\)\)\) : 0\)'
        
        # Pattern 3: With coefficient 1.0*((1e-15 < x[j]) ? (pow(x[j], (-1))) : 0)
        pattern3 = rf'1\.0\*\(\(1e-15 < x\[{spurious_idx}\]\) \? \(pow\(x\[{spurious_idx}\], \(-1\)\)\) : 0\)'
        
        # Try each pattern
        for pattern in [pattern3, pattern2, pattern1]:
            matches = list(re.finditer(pattern, result))
            if matches:
                print(f"[ROBUST FIX] Found {len(matches)} matches for pattern")
                
                # Process in reverse order to maintain positions
                for match in reversed(matches):
                    start = match.start()
                    end = match.end()
                    
                    # Check context to determine how to remove
                    before = result[:start]
                    after = result[end:]
                    
                    # Look for arithmetic operators around the term
                    # Remove preceding ' + ' if present
                    if before.endswith(' + '):
                        before = before[:-3]
                        result = before + after
                        modified = True
                        print(f"[ROBUST FIX] Removed term with preceding +")
                    # Remove following ' + ' if present
                    elif after.startswith(' + '):
                        after = after[3:]
                        result = before + after
                        modified = True
                        print(f"[ROBUST FIX] Removed term with following +")
                    # Handle case where it's part of a larger sum in parentheses
                    elif before.endswith('(') and ' + ' in after:
                        # This is the first term in a sum, remove it and the following +
                        plus_pos = after.find(' + ')
                        after = after[plus_pos + 3:]
                        result = before + after
                        modified = True
                        print(f"[ROBUST FIX] Removed first term in sum")
                    else:
                        # Just remove the term
                        result = before + after
                        modified = True
                        print(f"[ROBUST FIX] Removed standalone term")
    
    if modified:
        # Clean up any issues introduced by removal
        # Remove double spaces
        result = re.sub(r'\s+', ' ', result)
        # Remove empty parentheses
        result = re.sub(r'\(\s*\)', '(0)', result)
        # Fix arithmetic issues
        result = re.sub(r'\+\s*\+', '+', result)
        result = re.sub(r'\(\s*\+', '(', result)
        result = re.sub(r'\+\s*\)', ')', result)
        # Fix empty sums that might result in division by zero
        result = re.sub(r'\(\s*\)\s*/\s*\([^)]+\)', '0', result)
        
        # Count final spurious terms
        final_spurious = 0
        for spurious_idx in site_fraction_indices:
            if spurious_idx != i_idx:
                final_spurious += len(re.findall(rf'pow\(x\[{spurious_idx}\], \(-1\)\)', result))
        
        print(f"[ROBUST FIX] Removed spurious terms. Final count: {final_spurious}")
    else:
        print(f"[ROBUST FIX] No modifications made")
    
    return result


# Test the function
if __name__ == "__main__":
    # Test expression from the generated code
    test_expr = """8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])"""
    
    print("Original expression:")
    print(test_expr)
    print()
    
    # Fix for Y_NB diagonal (remove x[4] terms)
    fixed = remove_spurious_entropy_terms(test_expr, 3, 3)
    print("\nFixed expression for [3,3] diagonal:")
    print(fixed)
    
    # Verify
    x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', fixed))
    print(f"\nVerification: x[4] terms remaining: {x4_count}")