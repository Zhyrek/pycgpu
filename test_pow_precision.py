#!/usr/bin/env python
"""Test if pow() vs ** causes precision differences."""

import math
import numpy as np

# Test some typical values that might appear in thermodynamic calculations
test_cases = [
    (2.718281828, 2.5),  # e^2.5
    (10.0, -15.0),       # 10^-15 (like 1e-15)
    (0.898305085, 2.0),  # Site fraction squared
    (1.01694915, 3.0),   # Site fraction cubed
    (600.0, 0.5),        # Temperature square root
    (600.0, -1.0),       # 1/Temperature
]

print("COMPARING ** vs pow() PRECISION")
print("=" * 60)

for base, exp in test_cases:
    # Python ** operator
    result_op = base ** exp
    
    # math.pow function
    result_pow = math.pow(base, exp)
    
    # numpy power
    result_np = np.power(base, exp)
    
    # C-style computation (simulated)
    # In C, pow() from math.h is used
    
    diff_pow = abs(result_op - result_pow)
    diff_np = abs(result_op - result_np)
    
    print(f"\nbase={base}, exp={exp}")
    print(f"  Python **:   {result_op:.20e}")
    print(f"  math.pow():  {result_pow:.20e}")
    print(f"  numpy.power: {result_np:.20e}")
    print(f"  Diff (pow):  {diff_pow:.3e}")
    print(f"  Diff (np):   {diff_np:.3e}")
    
    if diff_pow > 1e-15:
        print("  WARNING: Significant difference!")

# Test a specific thermodynamic-like expression
print("\n" + "=" * 60)
print("THERMODYNAMIC EXPRESSION TEST")
print("=" * 60)

# Typical expression: R*T*log(x) where x is a site fraction
R = 8.314462618  # Gas constant
T = 600.0
x = 0.898305085  # Site fraction

# Expression: -R*T*x*log(x) + other terms
expr1_op = -R * T * x * math.log(x)
expr1_pow = -R * T * x * math.log(x)  # log doesn't use pow

# Expression with power: x**2 term
expr2_op = 1000.0 * x**2
expr2_pow = 1000.0 * math.pow(x, 2)

print(f"\nExpression 1: -R*T*x*log(x)")
print(f"  Result: {expr1_op:.15e}")

print(f"\nExpression 2: 1000*x**2") 
print(f"  Python **: {expr2_op:.15e}")
print(f"  math.pow:  {expr2_pow:.15e}")
print(f"  Diff:      {abs(expr2_op - expr2_pow):.3e}")

# Check if GPU is using float vs double precision
print("\n" + "=" * 60)
print("FLOAT vs DOUBLE PRECISION")
print("=" * 60)

x_double = 0.898305084745755
x_float = np.float32(x_double)

print(f"Double: {x_double:.15f}")
print(f"Float:  {x_float:.15f}")
print(f"Diff:   {abs(x_double - x_float):.3e}")

# Energy calculation at float precision
energy_double = -19941.00371858601
energy_float = np.float32(energy_double)

print(f"\nEnergy (double): {energy_double:.15f}")
print(f"Energy (float):  {energy_float:.15f}")
print(f"Diff:           {abs(energy_double - energy_float):.3e}")