#!/usr/bin/env python3
"""Test the fix_hessian_spurious_terms function"""

from pycalphad.gpu.gpu_codegen import fix_hessian_spurious_terms

# Test the function
test_str = '8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'
result = fix_hessian_spurious_terms(test_str, 3, 3)
print('Original:', test_str)
print('Fixed:', result)
print('Changed:', result != test_str)

# Test a more complex expression
complex_str = '-2.0*(x[3]*((x[2] < 2750.0) ? (-8519.353) : 0))/pow((x[3] + x[4]), 2) + 8.3145*x[2]*(1.0*((1e-15 < x[3]) ? (pow(x[3], (-1))) : 0) + 1.0*((1e-15 < x[4]) ? (pow(x[4], (-1))) : 0))/(x[3] + x[4])'
complex_result = fix_hessian_spurious_terms(complex_str, 3, 3)
print('\nComplex Original:', complex_str)
print('Complex Fixed:', complex_result)
print('Complex Changed:', complex_result != complex_str)