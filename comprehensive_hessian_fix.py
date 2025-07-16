#!/usr/bin/env python3
"""Create a comprehensive fix for the Hessian spurious terms"""

def fix_entropy_hessian_comprehensive(hess_code: str) -> str:
    """
    Comprehensive fix for entropy Hessian spurious terms.
    
    For a binary substitutional solution with entropy S = -R*T*sum(Y_i*log(Y_i))/(Y_1+Y_2),
    the correct Hessian should be:
    - Diagonal H[i,i] = R*T/Y_i (only one 1/Y_i term)
    - Off-diagonal H[i,j] = R*T (no 1/Y terms)
    
    The generated code has spurious terms from the (Y_1+Y_2) normalization.
    """
    import re
    
    lines = hess_code.split('\n')
    modified_lines = []
    total_removed = 0
    
    for line in lines:
        # Skip non-assignment lines
        if not re.match(r'\s*out\[\d+\]\s*=', line):
            modified_lines.append(line)
            continue
        
        # Extract the output index and expression
        match = re.match(r'(\s*)out\[(\d+)\]\s*=\s*(.+);', line)
        if not match:
            modified_lines.append(line)
            continue
            
        indent = match.group(1)
        out_idx = int(match.group(2))
        expression = match.group(3)
        
        # Map output indices to Hessian elements
        # For a 5-variable system (N, P, T, Y_NB, Y_TI):
        # Hessian is 5x5, stored in row-major order
        # out[18] = H[3,3] (Y_NB diagonal)
        # out[19] = H[3,4] (Y_NB,Y_TI off-diagonal)
        # out[23] = H[4,3] (Y_TI,Y_NB off-diagonal)
        # out[24] = H[4,4] (Y_TI diagonal)
        
        if out_idx == 18:  # H[3,3] - Y_NB diagonal
            print("Fixing H[3,3] (Y_NB diagonal)")
            # Should have only 1/x[3] terms, no 1/x[4] terms
            # Count and remove excess 1/x[3] and all 1/x[4]
            
            # Remove all 1/x[4] terms (spurious)
            x4_pattern = r'[+-]?\s*[\d.]*\*?\([^()]*\(1e-15 < x\[4\]\) \? \(pow\(x\[4\], \(-1\)\)\) : 0\)[^)]*'
            x4_matches = list(re.finditer(x4_pattern, expression))
            print(f"  Found {len(x4_matches)} spurious 1/x[4] terms")
            
            # Process in reverse to maintain positions
            for match in reversed(x4_matches):
                start = match.start()
                end = match.end()
                
                # Check for operators around the term
                before = expression[:start].rstrip()
                after = expression[end:].lstrip()
                
                # Remove the term and clean up operators
                if before.endswith('+'):
                    expression = before[:-1].rstrip() + ' ' + after
                elif before.endswith('-'):
                    expression = before[:-1].rstrip() + ' ' + after
                elif after.startswith('+'):
                    expression = before + after[1:].lstrip()
                elif after.startswith('-'):
                    expression = before + after
                else:
                    expression = before + after
                
                total_removed += 1
            
            # Now check if we have exactly one 1/x[3] term
            x3_count = len(re.findall(r'pow\(x\[3\], \(-1\)\)', expression))
            print(f"  After cleanup: {x3_count} 1/x[3] terms (should be 1)")
            
        elif out_idx == 24:  # H[4,4] - Y_TI diagonal
            print("Fixing H[4,4] (Y_TI diagonal)")
            # Should have only 1/x[4] terms, no 1/x[3] terms
            
            # Remove all 1/x[3] terms (spurious)
            x3_pattern = r'[+-]?\s*[\d.]*\*?\([^()]*\(1e-15 < x\[3\]\) \? \(pow\(x\[3\], \(-1\)\)\) : 0\)[^)]*'
            x3_matches = list(re.finditer(x3_pattern, expression))
            print(f"  Found {len(x3_matches)} spurious 1/x[3] terms")
            
            for match in reversed(x3_matches):
                start = match.start()
                end = match.end()
                
                before = expression[:start].rstrip()
                after = expression[end:].lstrip()
                
                if before.endswith('+'):
                    expression = before[:-1].rstrip() + ' ' + after
                elif before.endswith('-'):
                    expression = before[:-1].rstrip() + ' ' + after
                elif after.startswith('+'):
                    expression = before + after[1:].lstrip()
                elif after.startswith('-'):
                    expression = before + after
                else:
                    expression = before + after
                
                total_removed += 1
            
            x4_count = len(re.findall(r'pow\(x\[4\], \(-1\)\)', expression))
            print(f"  After cleanup: {x4_count} 1/x[4] terms (should be 1)")
            
        elif out_idx in [19, 23]:  # Off-diagonal elements
            print(f"Fixing H[{3 if out_idx==19 else 4},{4 if out_idx==19 else 3}] (off-diagonal)")
            # Should have NO 1/x terms at all
            
            # Remove all 1/x[3] and 1/x[4] terms
            for var_idx in [3, 4]:
                pattern = r'[+-]?\s*[\d.]*\*?\([^()]*\(1e-15 < x\[' + str(var_idx) + r'\]\) \? \(pow\(x\[' + str(var_idx) + r'\], \(-1\)\)\) : 0\)[^)]*'
                matches = list(re.finditer(pattern, expression))
                print(f"  Found {len(matches)} 1/x[{var_idx}] terms (all spurious)")
                
                for match in reversed(matches):
                    start = match.start()
                    end = match.end()
                    
                    before = expression[:start].rstrip()
                    after = expression[end:].lstrip()
                    
                    if before.endswith('+'):
                        expression = before[:-1].rstrip() + ' ' + after
                    elif before.endswith('-'):
                        expression = before[:-1].rstrip() + ' ' + after
                    elif after.startswith('+'):
                        expression = before + after[1:].lstrip()
                    elif after.startswith('-'):
                        expression = before + after
                    else:
                        expression = before + after
                    
                    total_removed += 1
        
        # Clean up the expression
        expression = re.sub(r'\s+', ' ', expression)
        expression = re.sub(r'\+\s*\+', '+', expression)
        expression = re.sub(r'-\s*-', '+', expression)
        expression = re.sub(r'\(\s*\)', '(0)', expression)
        expression = re.sub(r'\+\s*-', '-', expression)
        expression = expression.strip()
        
        # Reconstruct the line
        modified_lines.append(f"{indent}out[{out_idx}] = {expression};")
    
    print(f"\nTotal spurious terms removed: {total_removed}")
    return '\n'.join(modified_lines)


# Test on the generated code
if __name__ == "__main__":
    with open('generated_equilibrium_kernel.cu', 'r') as f:
        code = f.read()
    
    # Extract the Hessian function
    import re
    hess_match = re.search(r'(__device__ void pycgpu_model_0_formulahess\(double\* out, const double\* x\) \{.*?\n\})', code, re.DOTALL)
    
    if hess_match:
        hess_func = hess_match.group(1)
        fixed_hess = fix_entropy_hessian_comprehensive(hess_func)
        
        # Replace in the full code
        fixed_code = code.replace(hess_func, fixed_hess)
        
        # Save the fixed code
        with open('generated_equilibrium_kernel_fixed.cu', 'w') as f:
            f.write(fixed_code)
        
        print("\nFixed code saved to generated_equilibrium_kernel_fixed.cu")