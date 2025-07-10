#!/usr/bin/env python3
"""Compare CPU and GPU hessian expressions by evaluating them directly"""

import numpy as np
import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model
import re

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

# Get CPU phase record
cpu_phase_rec = prf.get('BCC_A2')

print("=== CPU Hessian ===")
# The CPU hessian is compiled, so we need to evaluate it
# Test point
dof = np.array([1.0, 101325.0, 1000.0, 0.612245, 0.387755])  # [N, P, T, Y_NB, Y_TI]
print(f"DOF values: {dof}")

# Evaluate CPU hessian
cpu_hess_result = np.zeros((5, 5), order='C')  # 5x5 matrix
cpu_phase_rec.formulahess(cpu_hess_result, dof)

# Extract element [3,3] (d²G/dY_NB²)
cpu_hess_33 = cpu_hess_result[3, 3]
print(f"CPU hessian[3,3] = {cpu_hess_33:.6f}")

# Also check other elements for comparison
cpu_hess_34 = cpu_hess_result[3, 4]  # [3,4]
cpu_hess_44 = cpu_hess_result[4, 4]  # [4,4]
print(f"CPU hessian[3,4] = {cpu_hess_34:.6f}")
print(f"CPU hessian[4,4] = {cpu_hess_44:.6f}")

print("\n=== GPU Hessian ===")
# Generate GPU hessian code
class MinimalWorkspace:
    def __init__(self, prf):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = prf
        
wks = MinimalWorkspace(prf)
gpu_hess_code = _nb_formulahess_from_model(model, 0, wks, validate=False, verbose=False)

# Extract the expression for element [3,3]
lines = gpu_hess_code.split('\n')
for line in lines:
    if 'out[18] =' in line:
        # This is element [3,3]
        expr = line.split('=', 1)[1].strip().rstrip(';')
        print(f"GPU hessian[3,3] expression (first 200 chars): {expr[:200]}...")
        
        # The expression is too complex to evaluate directly in Python
        # Let's compile and run it
        break

# Create a simple C program to evaluate the GPU expression
gpu_test_code = f"""
#include <stdio.h>
#include <math.h>

__device__ double pycgpu_model_0_formulahess_33(const double* x) {{
    // Element [3,3] from the GPU code
    return {expr};
}}

// Host wrapper
double evaluate_gpu_hessian_33(const double* x) {{
    return {expr.replace('__device__', '').replace('pow', 'pow')};
}}

int main() {{
    double x[5] = {{1.0, 101325.0, 1000.0, 0.612245, 0.387755}};
    double result = evaluate_gpu_hessian_33(x);
    printf("GPU hessian[3,3] = %.6f\\n", result);
    return 0;
}}
"""

# Save and compile the test
with open('test_gpu_hess.c', 'w') as f:
    f.write(gpu_test_code)

print("\n=== Attempting to evaluate GPU expression ===")
# This is complex, let's try a different approach

# Parse the GPU expression to understand its structure
print("\n=== Analyzing GPU Expression Structure ===")

# Count occurrences of key patterns
gpu_expr = expr
count_div_sum = gpu_expr.count('/(x[3] + x[4])')
count_div_sum2 = gpu_expr.count('/pow((x[3] + x[4]), 2)')
count_div_sum3 = gpu_expr.count('/pow((x[3] + x[4]), 3)')

print(f"Occurrences of /(x[3] + x[4]): {count_div_sum}")
print(f"Occurrences of /pow((x[3] + x[4]), 2): {count_div_sum2}")
print(f"Occurrences of /pow((x[3] + x[4]), 3): {count_div_sum3}")

# Let's manually evaluate a simplified version
print("\n=== Manual Evaluation of Key Terms ===")
x = dof
T = x[2]
Y_NB = x[3]
Y_TI = x[4]
sum_Y = Y_NB + Y_TI

# From the GPU expression, we can identify some key terms:
# 1. Ideal entropy: 8.3145*T/Y_NB
ideal_term = 8.3145 * T / Y_NB
print(f"Ideal entropy term: 8.3145*T/Y_NB = {ideal_term:.2f}")

# 2. Terms with (Y_NB + Y_TI) denominators
# These are the spurious terms that shouldn't exist

# The CPU hessian should primarily be the ideal entropy term
# plus any contributions from excess terms

print(f"\n=== Comparison ===")
print(f"CPU hessian[3,3] = {cpu_hess_33:.2f}")
print(f"Expected (ideal only) = {ideal_term:.2f}")
print(f"Difference = {cpu_hess_33 - ideal_term:.2f}")
print(f"\nRatio GPU/CPU would be: ~2.6 based on previous analysis")

# Let's also check the gradient to ensure it's working
print("\n=== Checking Gradient ===")
grad_result = np.zeros(5)
cpu_phase_rec.formulagrad(grad_result, dof)
print(f"CPU gradient: {grad_result}")