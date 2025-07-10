#!/usr/bin/env python3
"""Find the CPU's secret to avoiding spurious terms"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions, sympify, lambdify
import symengine as se
import numpy as np

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

print("=== The Final Test ===")

# Get the exact symbolic hessian the CPU computes
variables = prf.state_variables + model.site_fractions
wrt = sympify(tuple(variables))
graph = sympify(model.G)

# Get hessian element [3,3]
grad_Y_NB = graph.diff(wrt[3])
hess_33_symbolic = grad_Y_NB.diff(wrt[3])

print(f"Symbolic hessian[3,3] has {len(str(hess_33_symbolic))} characters")

# Now lambdify it exactly as the CPU does
from pycalphad.codegen.sympydiff_utils import _get_lambidfy_options
inp = sympify(variables)
hess_func = lambdify(inp, [hess_33_symbolic])

# Test it
test_vals = [1.0, 101325.0, 1000.0, 0.6, 0.4]
cpu_result = hess_func(test_vals)
print(f"\nCPU lambdified result: {cpu_result:.2f}")
print(f"Expected (RT/Y_NB): {8.3145 * 1000 / 0.6:.2f}")

# Now test with sum != 1
test_vals_09 = [1.0, 101325.0, 1000.0, 0.6, 0.3]
cpu_result_09 = hess_func(test_vals_09)
print(f"\nWith sum=0.9: {cpu_result_09:.2f}")

# The REAL test: Let's manually create the spurious term and see what happens
print("\n=== Testing Spurious Term ===")

# Create the spurious term that appears in GPU: RT*(1/Y_NB + 1/Y_TI)/(Y_NB+Y_TI)
Y_NB = wrt[3]
Y_TI = wrt[4]
T = wrt[2]
R = 8.3145

spurious = R * T * (1/Y_NB + 1/Y_TI) / (Y_NB + Y_TI)
spurious_func = lambdify(inp, [spurious])

spur_val = spurious_func(test_vals)
print(f"\nSpurious term at sum=1.0: {spur_val:.2f}")
print(f"This equals: RT/Y_NB + RT/Y_TI = {8.3145*1000/0.6 + 8.3145*1000/0.4:.2f}")

spur_val_09 = spurious_func(test_vals_09)
print(f"\nSpurious term at sum=0.9: {spur_val_09:.2f}")

# Check if the hessian contains this spurious term
if 'BCC_A20NB**(-1) + BCC_A20TI**(-1)' in str(hess_33_symbolic) or \
   'BCC_A20NB**(-1.0) + BCC_A20TI**(-1.0)' in str(hess_33_symbolic):
    print("\nFOUND: The symbolic hessian DOES contain (1/Y_NB + 1/Y_TI)!")
    
    # Extract the relevant part
    hess_str = str(hess_33_symbolic)
    if '8.3145*T*' in hess_str:
        idx = hess_str.find('8.3145*T*')
        context = hess_str[idx:idx+200]
        print(f"Context: {context}")

print("\n=== The Answer ===")
print("We need to check if the CPU's symbolic hessian actually has the spurious term")
print("but it somehow cancels out during evaluation...")