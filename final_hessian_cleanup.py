#!/usr/bin/env python3
"""Apply final cleanup to remove spurious terms from generated Hessian code"""

import re

def cleanup_hessian_spurious_terms(full_code):
    """
    Final cleanup pass to remove spurious entropy terms from the generated Hessian.
    This operates on the complete generated code to catch any terms that slipped through.
    """
    lines = full_code.split('\n')
    modified_lines = []
    
    for line in lines:
        # Check if this is a Hessian output line
        match = re.match(r'\s*out\[(\d+)\]\s*=\s*(.+);', line)
        if match:
            out_idx = int(match.group(1))
            expression = match.group(2)
            
            # Map output index to i,j indices for 5x5 Hessian
            # out[k] = hess[i,j] where k = i*5 + j
            n = 5  # 5 variables: N, P, T, Y_NB, Y_TI
            i = out_idx // n
            j = out_idx % n
            
            # Only process diagonal elements for site fractions
            if i == j and i >= 3:  # Indices 3 and 4 are Y_NB and Y_TI
                print(f"[FINAL CLEANUP] Processing out[{out_idx}] = hess[{i},{j}]")
                
                # Count spurious terms before
                spurious_indices = [3, 4]
                spurious_indices.remove(i)  # Don't remove the correct diagonal term
                
                before_count = 0
                for idx in spurious_indices:
                    before_count += len(re.findall(rf'pow\(x\[{idx}\], \(-1\)\)', expression))
                
                if before_count > 0:
                    print(f"[FINAL CLEANUP] Found {before_count} spurious terms")
                    
                    # Remove spurious terms
                    for spurious_idx in spurious_indices:
                        # Pattern for entropy terms with spurious 1/x[j]
                        patterns = [
                            # Most specific: coefficient * conditional
                            rf'1\.0\*\(\(1e-15 < x\[{spurious_idx}\]\) \? \(pow\(x\[{spurious_idx}\], \(-1\)\)\) : 0\)',
                            # Just the conditional
                            rf'\(\(1e-15 < x\[{spurious_idx}\]\) \? \(pow\(x\[{spurious_idx}\], \(-1\)\)\) : 0\)',
                            # Simple pow
                            rf'pow\(x\[{spurious_idx}\], \(-1\)\)'
                        ]
                        
                        for pattern in patterns:
                            # Find all occurrences
                            while True:
                                match = re.search(pattern, expression)
                                if not match:
                                    break
                                    
                                start = match.start()
                                end = match.end()
                                before = expression[:start]
                                after = expression[end:]
                                
                                # Remove with appropriate handling of operators
                                if before.endswith(' + ') and after:
                                    expression = before[:-3] + after
                                elif before and after.startswith(' + '):
                                    expression = before + after[3:]
                                elif before.endswith('(') and after.startswith(')'):
                                    # Removing the only term in parentheses
                                    expression = before + '0' + after
                                else:
                                    expression = before + after
                    
                    # Clean up
                    expression = re.sub(r'\s+', ' ', expression)
                    expression = re.sub(r'\+\s*\+', '+', expression)
                    expression = re.sub(r'\(\s*\)', '(0)', expression)
                    expression = re.sub(r'\(\s*\+', '(', expression)
                    expression = re.sub(r'\+\s*\)', ')', expression)
                    
                    # Count after
                    after_count = 0
                    for idx in spurious_indices:
                        after_count += len(re.findall(rf'pow\(x\[{idx}\], \(-1\)\)', expression))
                    
                    print(f"[FINAL CLEANUP] Removed {before_count - after_count} spurious terms")
                
                line = f"    out[{out_idx}] = {expression};"
        
        modified_lines.append(line)
    
    return '\n'.join(modified_lines)


if __name__ == "__main__":
    # Test on generated code
    with open('generated_equilibrium_kernel.cu', 'r') as f:
        code = f.read()
    
    cleaned = cleanup_hessian_spurious_terms(code)
    
    with open('generated_equilibrium_kernel_cleaned.cu', 'w') as f:
        f.write(cleaned)
    
    print("\nCleaned code saved to generated_equilibrium_kernel_cleaned.cu")
    
    # Verify the fix
    print("\n=== Verification ===")
    for idx in [18, 24]:  # Y_NB and Y_TI diagonals
        original_count = len(re.findall(rf'out\[{idx}\][^;]+pow\(x\[4\], \(-1\)\)', code))
        cleaned_count = len(re.findall(rf'out\[{idx}\][^;]+pow\(x\[4\], \(-1\)\)', cleaned))
        print(f"out[{idx}]: pow(x[4], (-1)) count - original: {original_count}, cleaned: {cleaned_count}")