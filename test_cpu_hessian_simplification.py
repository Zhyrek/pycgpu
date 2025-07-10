#!/usr/bin/env python3
"""Test if CPU simplifies site fraction sum during differentiation"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions
import symengine as se

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

print("=== Analyzing CPU Hessian Generation ===")
print(f"Model G expression (first 500 chars): {str(model.G)[:500]}...")

# Get the variables for differentiation
variables = prf.state_variables + model.site_fractions
print(f"\nVariables for differentiation: {variables}")

# Manually check the G expression structure
G_str = str(model.G)
if 'BCC_A20NB + BCC_A20TI' in G_str:
    print("\nG expression contains (Y_NB + Y_TI) factor")
    
# Build the hessian using CPU's approach
print("\n=== Building Hessian with CPU's build_functions ===")
funcs = build_functions(model.G, variables, include_obj=True, include_grad=True, include_hess=True)

print(f"\nFunctions returned: {type(funcs)}")

# The CPU might have internal simplifications
# Let's check if we can access the intermediate symbolic expressions
print("\n=== Checking for Site Fraction Sum Simplification ===")

# Test a simple symbolic expression to understand the behavior
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI')
T = se.Symbol('T')

# Create a simple test expression similar to the model
test_expr = (Y_NB + Y_TI) * (100 * Y_NB + 200 * Y_TI + 50 * T)

print(f"\nTest expression: {test_expr}")

# Differentiate twice with respect to Y_NB
d1 = test_expr.diff(Y_NB)
d2 = d1.diff(Y_NB)

print(f"\nFirst derivative d/dY_NB: {d1}")
print(f"Second derivative d²/dY_NB²: {d2}")

# Check if it contains (Y_NB + Y_TI) in denominator
d2_str = str(d2)
if '/(Y_NB + Y_TI)' in d2_str or '(Y_NB + Y_TI)**(-' in d2_str:
    print("\nSecond derivative contains (Y_NB + Y_TI) in denominator!")
else:
    print("\nSecond derivative does NOT contain (Y_NB + Y_TI) in denominator")

# Now let's see what happens if we substitute Y_NB + Y_TI = 1
print("\n=== Testing with Constraint Y_NB + Y_TI = 1 ===")
# Express Y_TI = 1 - Y_NB
test_expr_constrained = test_expr.subs(Y_TI, 1 - Y_NB)
print(f"\nConstrained expression: {test_expr_constrained}")

d1_c = test_expr_constrained.diff(Y_NB)
d2_c = d1_c.diff(Y_NB)

print(f"\nFirst derivative (constrained): {d1_c}")
print(f"Second derivative (constrained): {d2_c}")

# This should give us a constant value without (Y_NB + Y_TI) terms