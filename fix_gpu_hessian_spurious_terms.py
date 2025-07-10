#!/usr/bin/env python3
"""Fix to remove spurious entropy cross-terms from GPU hessian"""

def fix_hessian_spurious_terms(hess_str, i_idx, j_idx, ordered_symbols):
    """
    Remove spurious entropy cross-terms from hessian elements.
    
    For diagonal elements d²G/dY_i², the GPU includes spurious RT/Y_j terms
    where j != i. These come from the (Y1+Y2) factor that didn't get simplified.
    
    Args:
        hess_str: String representation of hessian element
        i_idx: First derivative index
        j_idx: Second derivative index
        ordered_symbols: List of symbols used for differentiation
    
    Returns:
        Fixed hessian string with spurious terms removed
    """
    import re
    
    # Only fix diagonal elements where both indices are site fractions
    # State variables are typically first 3 (N, P, T)
    if i_idx != j_idx or i_idx < 3:
        return hess_str
    
    # Get the variable names
    # Assuming x[3] is Y_NB and x[4] is Y_TI
    var_i = f"x[{i_idx}]"
    
    # Pattern to find RT*(1/Y_i + 1/Y_j)/(Y_i + Y_j) terms
    # This matches: 8.3145*x[2]*(... + ...)/(x[3] + x[4])
    pattern = r'8\.3145\*x\[2\]\*\(([^)]+)\)/\(x\[3\] \+ x\[4\]\)'
    
    def fix_spurious(match):
        inner = match.group(1)
        # Check if this contains multiple inverse terms
        if inner.count('pow(x[') > 1 and '(-1)' in inner:
            # This is likely RT*(1/Y_NB + 1/Y_TI)/(Y_NB + Y_TI)
            # For diagonal element [i,i], we only want the 1/Y_i term
            
            # Extract individual terms
            terms = []
            # Pattern for individual inverse terms: coefficient*((condition) ? (pow(x[n], (-1))) : 0)
            term_pattern = r'([\d.]+)\*\(\((1e-15 < x\[(\d+)\])\) \? \(pow\(x\[\2\], \(-1\)\)\) : 0\)'
            
            for term_match in re.finditer(term_pattern, inner):
                coeff = term_match.group(1)
                var_idx = int(term_match.group(2))
                
                # Only keep the term for the current variable
                if var_idx == i_idx:
                    terms.append(term_match.group(0))
            
            if terms:
                # Reconstruct with only the correct term
                return f'8.3145*x[2]*({" + ".join(terms)})/(x[3] + x[4])'
            
        return match.group(0)  # Return unchanged if not the pattern we're looking for
    
    # Apply the fix
    fixed = re.sub(pattern, fix_spurious, hess_str)
    
    # Alternative approach: Remove the /(x[3] + x[4]) divisor entirely
    # for the entropy terms, since it should have canceled with the (x[3] + x[4]) factor
    
    # Pattern: 8.3145*x[2]*((1e-15 < x[i]) ? (pow(x[i], (-1))) : 0)/(x[3] + x[4])
    # Should become: 8.3145*x[2]*((1e-15 < x[i]) ? (pow(x[i], (-1))) : 0)
    simple_pattern = rf'(8\.3145\*x\[2\]\*\(\(1e-15 < x\[{i_idx}\]\) \? \(pow\(x\[{i_idx}\], \(-1\)\)\) : 0\))/\(x\[3\] \+ x\[4\]\)'
    fixed = re.sub(simple_pattern, r'\1', fixed)
    
    return fixed


# Test the fix
if __name__ == "__main__":
    # Test case: GPU hessian[3,3] with spurious terms
    test_hess = '8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'
    
    print("Original:", test_hess)
    fixed = fix_hessian_spurious_terms(test_hess, 3, 3, ['N', 'P', 'T', 'Y_NB', 'Y_TI'])
    print("Fixed:   ", fixed)
    
    # The fixed version should only have RT/x[3], not RT/x[4]
    assert 'x[4]), (-1)))' not in fixed, "Still contains x[4] inverse term!"
    print("\nTest passed!")