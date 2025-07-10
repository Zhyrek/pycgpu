#!/usr/bin/env python3
"""Analyze the generated Hessian code to find the extra factor"""

# Read the generated code
with open('generated_cuda_code.cu', 'r') as f:
    code = f.read()

# Find the formulahess function
import re
match = re.search(r'__device__ void pycgpu_model_0_formulahess\(double\* out, const double\* x\) \{([^}]+)\}', code, re.DOTALL)

if match:
    func_body = match.group(1)
    
    # Find out[18] which is hess[3,3]
    out18_match = re.search(r'out\[18\] = ([^;]+);', func_body)
    if out18_match:
        out18_expr = out18_match.group(1)
        
        print("=== Analyzing out[18] (hess[3,3]) ===\n")
        print(f"Expression length: {len(out18_expr)} characters\n")
        
        # Count different types of terms
        pow_x3_count = out18_expr.count('pow(x[3], (-1))')
        pow_x4_count = out18_expr.count('pow(x[4], (-1))')
        
        print(f"pow(x[3], (-1)) occurrences: {pow_x3_count}")
        print(f"pow(x[4], (-1)) occurrences: {pow_x4_count}")
        
        # Check for terms divided by (x[3] + x[4])
        div_sum_count = out18_expr.count('/(x[3] + x[4])')
        div_sum_pow2_count = out18_expr.count('/pow((x[3] + x[4]), 2)')
        div_sum_pow3_count = out18_expr.count('/pow((x[3] + x[4]), 3)')
        
        print(f"\nDivision by (x[3] + x[4]): {div_sum_count}")
        print(f"Division by (x[3] + x[4])²: {div_sum_pow2_count}")
        print(f"Division by (x[3] + x[4])³: {div_sum_pow3_count}")
        
        # Check entropy terms
        entropy_terms = re.findall(r'8\.3145\*x\[2\]\*[^;]+?/(x\[3\] \+ x\[4\])', out18_expr)
        print(f"\nEntropy-related terms (8.3145*T*...): {len(entropy_terms)}")
        
        # The fix should have removed pow(x[4], (-1)) from diagonal elements
        if pow_x4_count > 0:
            print("\nERROR: Spurious pow(x[4], (-1)) terms still present!")
        else:
            print("\nGOOD: No spurious pow(x[4], (-1)) terms found.")
            
        print("\nThe 2.5x factor might come from:")
        print("1. Multiple divisions by (x[3] + x[4]) = 1.0")
        print("2. Remaining pow(x[3], (-1)) terms")
        print("3. Different normalization in the energy expression")
        
else:
    print("Could not find formulahess function")