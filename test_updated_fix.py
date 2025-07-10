#!/usr/bin/env python3
"""Test the updated fix_hessian_spurious_terms function"""

from pycalphad.gpu.gpu_codegen import fix_hessian_spurious_terms

# Test with a complex expression from the actual generated code
# This is a simplified version of what appears in out[18]
test_expr = (
    "8.3145*x[0]*(1.0*((1e-15 < x[2]) ? (pow(x[2], (-1))) : 0) + "
    "1.0*((1e-15 < x[1]) ? (pow(x[1], (-1))) : 0))/(x[1] + x[2]) + "
    "16.629*x[0]*(1.0*((1e-15 < x[2]) ? (pow(x[2], (-1))) : 0) + "
    "1.0*((1e-15 < x[1]) ? (1 + log(x[1])) : 0))/(x[1] + x[2])"
)

print("Testing fix for Y_NB diagonal (i_idx=1, j_idx=1)")
print("Original expression:")
print(test_expr)
print("\nApplying fix...")

# Test for Y_NB diagonal (should remove x[2] terms)
fixed = fix_hessian_spurious_terms(test_expr, 1, 1)

print("\nFixed expression:")
print(fixed)

print("\n" + "="*60 + "\n")

# Test for Y_TI diagonal (should remove x[1] terms)
print("Testing fix for Y_TI diagonal (i_idx=2, j_idx=2)")
fixed2 = fix_hessian_spurious_terms(test_expr, 2, 2)

print("\nFixed expression:")
print(fixed2)