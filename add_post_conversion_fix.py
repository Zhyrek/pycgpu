#!/usr/bin/env python3
"""Add the post-conversion fix function to gpu_codegen.py"""

# Read the file
with open('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py', 'r') as f:
    content = f.read()

# Find where to insert the new function - after fix_hessian_spurious_terms_v2
insert_pos = content.find('def fix_hessian_spurious_terms(')
if insert_pos == -1:
    print("Could not find fix_hessian_spurious_terms function")
    exit(1)

# Find the end of fix_hessian_spurious_terms_v2 (just before fix_hessian_spurious_terms)
# Go backwards from insert_pos to find the previous function
func_start = insert_pos - 1
while func_start > 0 and content[func_start] != '\n':
    func_start -= 1

# Insert the new function
new_function = '''

def fix_hessian_spurious_terms_post_conversion(hess_str, i_idx, j_idx, num_statevars=3):
    """
    Remove spurious entropy cross-terms from diagonal hessian elements AFTER variable conversion.
    
    For d²G/dx[i]², remove RT/x[j] terms where j != i.
    This works on the converted x[i] format rather than variable names.
    
    The spurious terms come from the (Y_NB + Y_TI) denominator in the entropy expression.
    When differentiated twice, it creates cross-terms that shouldn't be in the diagonal.
    """
    import re
    
    # Only process diagonal elements
    if i_idx != j_idx:
        return hess_str
        
    # Only process site fraction indices (after state variables)
    if i_idx < num_statevars:
        return hess_str
    
    # For diagonal element x[i], we need to remove 1/x[j] terms where j != i
    # These appear in entropy expressions like:
    # 8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])
    
    # Find all indices that appear in 1/x[j] terms
    inv_pattern = r'1\.0\*\(\(1e-15 < x\[(\d+)\]\) \? \(pow\(x\[\d+\], \(-1\)\)\) : 0\)'
    
    modified = False
    fixed = hess_str
    
    # Find all such terms
    matches = list(re.finditer(inv_pattern, hess_str))
    
    # Process in reverse order to maintain string positions
    for match in reversed(matches):
        idx_str = match.group(1)
        idx = int(idx_str)
        
        # If this is a different index than our diagonal, it's spurious
        if idx != i_idx and idx >= num_statevars:
            # Remove this term
            term_to_remove = match.group(0)
            
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
    
    if modified:
        # Clean up any double spaces or arithmetic issues
        fixed = re.sub(r'\s+', ' ', fixed)
        fixed = re.sub(r'\(\s*\+', '(', fixed)
        fixed = re.sub(r'\+\s*\)', ')', fixed)
        fixed = re.sub(r'\+\s*\+', '+', fixed)
        
        # Special case: if we removed all terms from a sum, we might have just 0
        # e.g., "8.3145*x[2]*(0)/(x[3] + x[4])" -> "0"
        fixed = re.sub(r'[\d.]+\*[^*]+\*\(0\)/[^)]+', '0', fixed)
        
        # Count how many terms were removed
        original_inv_count = len(re.findall(r'pow\(x\[\d+\], \(-1\)\)', hess_str))
        final_inv_count = len(re.findall(r'pow\(x\[\d+\], \(-1\)\)', fixed))
        print(f"[GPU HESSIAN FIX POST] Diagonal x[{i_idx}]: Removed {original_inv_count - final_inv_count} spurious 1/x[j] terms")
    
    return fixed

'''

# Insert the new function
new_content = content[:func_start] + new_function + content[func_start:]

# Write back
with open('/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/gpu_codegen.py', 'w') as f:
    f.write(new_content)

print("Added fix_hessian_spurious_terms_post_conversion function")