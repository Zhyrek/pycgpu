#!/usr/bin/env python3
"""Test the fix with workspace indices"""

from pycalphad.gpu.gpu_codegen import fix_hessian_spurious_terms

# Test with workspace indices - this is what appears in the actual generated code
# In workspace: N=x[0], P=x[1], T=x[2], Y_NB=x[3], Y_TI=x[4]
test_expr = (
    "8.3145*x[2]*(1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0) + "
    "1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0))/(x[3] + x[4]) + "
    "16.629*x[2]*(1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0) + "
    "1.0*((1e-15 < x[3]) ? (1 + log(x[3])) : 0))/(x[3] + x[4])"
)

print("Testing fix for Y_NB diagonal with workspace indices")
print("Original expression:")
print(test_expr)
print("\nApplying fix...")

# Test for Y_NB diagonal (i_idx=1 in model space, should remove x[4] terms)
fixed = fix_hessian_spurious_terms(test_expr, 1, 1)

print("\nFixed expression:")
print(fixed)

print("\n" + "="*60 + "\n")

# Test for Y_TI diagonal (i_idx=2 in model space, should remove x[3] terms)
print("Testing fix for Y_TI diagonal with workspace indices")
fixed2 = fix_hessian_spurious_terms(test_expr, 2, 2)

print("\nFixed expression:")
print(fixed2)