#!/usr/bin/env python3
"""Evaluate GPU hessian expression directly by parsing key terms"""

import numpy as np

# Test values
x = [1.0, 101325.0, 1000.0, 0.612245, 0.387755]  # [N, P, T, Y_NB, Y_TI]
T = x[2]
Y_NB = x[3]
Y_TI = x[4]

print(f"=== Test Values ===")
print(f"T = {T}")
print(f"Y_NB = {Y_NB}")
print(f"Y_TI = {Y_TI}")
print(f"Y_NB + Y_TI = {Y_NB + Y_TI}")

# CPU result
cpu_hess_33 = 13580.347737
print(f"\nCPU hessian[3,3] = {cpu_hess_33}")

# Manually evaluate key terms from the GPU expression
# The GPU expression for element [3,3] has many terms. Let's evaluate the main ones:

# Energy values at T=1000K (from the Piecewise expressions in the code)
energy_NB = -8519.353 + 142.045475*T - 26.4711*T*np.log(T) + 93399.0/T + 0.000203475*T**2 - 3.5012e-07*T**3
energy_TI = -1272.064 + 134.71418*T - 25.5768*T*np.log(T) + 7208.0/T - 0.000663845*T**2 - 2.78803e-07*T**3

print(f"\n=== Energy Values ===")
print(f"energy_NB = {energy_NB}")
print(f"energy_TI = {energy_TI}")

# Now let's evaluate the main terms from the GPU expression
sum_Y = Y_NB + Y_TI

# Term 1: -2.0*(Y_NB*energy_NB + Y_TI*energy_TI)/pow((Y_NB + Y_TI), 2)
term1 = -2.0 * (Y_NB * energy_NB + Y_TI * energy_TI) / (sum_Y**2)

# Term 2: 26090.6*Y_TI/(Y_NB + Y_TI)
L = 26090.6
term2 = L * Y_TI / sum_Y

# Term 3: 2.0*energy_NB/(Y_NB + Y_TI)
term3 = 2.0 * energy_NB / sum_Y

# Term 4: There's a complex term with (Y_NB + Y_TI) factor
# 1.0*(Y_NB + Y_TI)*(2.0*(Y_NB*energy_NB + Y_TI*energy_TI)/pow((Y_NB + Y_TI), 3) - ...)
# This simplifies to: 2.0*(Y_NB*energy_NB + Y_TI*energy_TI)/pow((Y_NB + Y_TI), 2)
term4_part1 = 2.0 * (Y_NB * energy_NB + Y_TI * energy_TI) / (sum_Y**2)

# Term 5: -26090.6*Y_TI/pow((Y_NB + Y_TI), 2)
term5 = -L * Y_TI / (sum_Y**2)

# Term 6: -2.0*energy_NB/pow((Y_NB + Y_TI), 2)
term6 = -2.0 * energy_NB / (sum_Y**2)

# Entropy terms
R = 8.3145
# From the expression: 8.3145*T/Y_NB (appears in multiple places)
entropy_term = R * T / Y_NB

# There are also logarithmic terms
# 16.629*T*(Y_NB*log(Y_NB) + Y_TI*log(Y_TI))/pow((Y_NB + Y_TI), 3)
log_term1 = 16.629 * T * (Y_NB * np.log(Y_NB) + Y_TI * np.log(Y_TI)) / (sum_Y**3)

# -16.629*T*(1 + log(Y_NB))/pow((Y_NB + Y_TI), 2)
log_term2 = -16.629 * T * (1 + np.log(Y_NB)) / (sum_Y**2)

# And more entropy-related terms
# The key entropy contribution should be: 8.3145*T/Y_NB
# But there are also terms like: 8.3145*T*(1/Y_NB + 1/Y_TI)/(Y_NB + Y_TI)
mixed_entropy = R * T * (1/Y_NB + 1/Y_TI) / sum_Y

# Interaction terms  
# 26090.6*Y_NB*Y_TI/pow((Y_NB + Y_TI), 3)
interaction_term1 = L * Y_NB * Y_TI / (sum_Y**3)

# -26090.6*Y_NB*Y_TI/pow((Y_NB + Y_TI), 2)
interaction_term2 = -L * Y_NB * Y_TI / (sum_Y**2)

print(f"\n=== GPU Expression Terms ===")
print(f"Term 1: -2*(Y_NB*g_NB + Y_TI*g_TI)/(Y_NB+Y_TI)^2 = {term1:.2f}")
print(f"Term 2: L*Y_TI/(Y_NB+Y_TI) = {term2:.2f}")
print(f"Term 3: 2*g_NB/(Y_NB+Y_TI) = {term3:.2f}")
print(f"Term 4: 2*(Y_NB*g_NB + Y_TI*g_TI)/(Y_NB+Y_TI)^2 = {term4_part1:.2f}")
print(f"Term 5: -L*Y_TI/(Y_NB+Y_TI)^2 = {term5:.2f}")
print(f"Term 6: -2*g_NB/(Y_NB+Y_TI)^2 = {term6:.2f}")
print(f"\nEntropy term: RT/Y_NB = {entropy_term:.2f}")
print(f"Log term 1: {log_term1:.2f}")
print(f"Log term 2: {log_term2:.2f}")
print(f"Mixed entropy: RT*(1/Y_NB + 1/Y_TI)/(Y_NB+Y_TI) = {mixed_entropy:.2f}")
print(f"Interaction term 1: L*Y_NB*Y_TI/(Y_NB+Y_TI)^3 = {interaction_term1:.2f}")
print(f"Interaction term 2: -L*Y_NB*Y_TI/(Y_NB+Y_TI)^2 = {interaction_term2:.2f}")

# Try to sum up the main contributions
# Note: This is a simplified evaluation - the actual expression has many more terms
estimated_gpu = entropy_term  # Start with the main entropy term

# Add some of the spurious terms that shouldn't be there
# These come from not simplifying (Y_NB + Y_TI) = 1
estimated_gpu += term1 + term2 + term3 + term4_part1 + term5 + term6
estimated_gpu += interaction_term1 + interaction_term2

print(f"\n=== Comparison ===")
print(f"CPU hessian[3,3] = {cpu_hess_33:.2f}")
print(f"Estimated GPU (partial) = {estimated_gpu:.2f}")
print(f"Ratio (partial estimate) = {estimated_gpu/cpu_hess_33:.3f}")

print(f"\n=== Analysis ===")
print("The GPU expression contains many spurious terms with (Y_NB + Y_TI) in denominators.")
print("These terms arise from differentiating without first simplifying (Y_NB + Y_TI) = 1.")
print("When Y_NB + Y_TI = 1, these denominators don't cause division by zero,")
print("but they create extra contributions that shouldn't exist.")