#!/usr/bin/env python3
"""Trace exact difference in code generation between CPU and GPU"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.gpu.gpu_codegen import _nb_formulahess_from_model, notebook_source_from_expr
import symengine as se

# Load model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

# Create phase record factory
conditions = {v.T: 1000, v.P: 101325, v.N: 1}
prf = PhaseRecordFactory(db, ['NB', 'TI', 'VA'], conditions, {'BCC_A2': model})

print("=== Model G Expression ===")
print(f"Type: {type(model.G)}")
print(f"First 300 chars: {str(model.G)[:300]}...")

# Extract the ideal entropy part
G_str = str(model.G)
import re
entropy_match = re.search(r'8\.3145\*T\*\(.*?\)\/\(BCC_A20NB \+ BCC_A20TI\)', G_str)
if entropy_match:
    print(f"\nEntropy term found: {entropy_match.group()[:100]}...")

print("\n=== CPU Hessian Generation ===")
# CPU uses build_functions
variables = prf.state_variables + model.site_fractions
print(f"Variables for differentiation: {variables}")

# Let's manually trace what build_functions does
print("\n--- Tracing build_functions ---")
from pycalphad.codegen.sympydiff_utils import sympify
import symengine

# This is what build_functions does:
wrt = sympify(tuple(variables))
inp = sympify(variables)
graph = sympify(model.G)

print(f"graph type: {type(graph)}")
print(f"wrt: {wrt}")

# Check the first derivative
print("\n--- First Derivatives ---")
grad_graphs = list(graph.diff(w) for w in wrt)
# Check the gradient w.r.t. Y_NB (index 3)
print(f"dG/dY_NB type: {type(grad_graphs[3])}")
print(f"dG/dY_NB (first 200 chars): {str(grad_graphs[3])[:200]}...")

# Check for 1/Y terms in gradient
grad_str = str(grad_graphs[3])
if '**(-1)' in grad_str or '**(-1.0)' in grad_str:
    print("WARNING: Found 1/Y terms in gradient!")
else:
    print("Good: No 1/Y terms in gradient")

# Check the second derivative
print("\n--- Second Derivatives ---")
hess_33 = grad_graphs[3].diff(wrt[3])
print(f"d²G/dY_NB² type: {type(hess_33)}")
print(f"d²G/dY_NB² (first 200 chars): {str(hess_33)[:200]}...")

# Now let's see what the GPU does
print("\n\n=== GPU Hessian Generation ===")

# Create minimal workspace
class MinimalWorkspace:
    def __init__(self, prf):
        self.components = ['NB', 'TI', 'VA']
        self.phase_record_factory = prf
        
wks = MinimalWorkspace(prf)

# This is what the GPU does
print("--- Tracing notebook_source_from_expr ---")
ordered_symbols = [v.N, v.P, v.T, model.site_fractions[0], model.site_fractions[1]]
print(f"GPU ordered symbols: {ordered_symbols}")

# The GPU differentiates model.G
gpu_graph = model.G
print(f"GPU graph same as CPU? {gpu_graph == graph}")

# Check GPU first derivatives
gpu_grad_3 = gpu_graph.diff(ordered_symbols[3])
print(f"\nGPU dG/dY_NB same as CPU? {gpu_grad_3 == grad_graphs[3]}")

# The issue might be in the conversion to C code
print("\n--- Checking C Code Generation ---")

# Let's generate a simple test case
Y_NB = se.Symbol('Y_NB')
Y_TI = se.Symbol('Y_TI') 
T = se.Symbol('T')

# Simple entropy term
S_test = 8.3145 * T * (Y_NB * se.log(Y_NB) + Y_TI * se.log(Y_TI))
print(f"\nTest expression: {S_test}")

# First derivative
dS_dY_NB = S_test.diff(Y_NB)
print(f"dS/dY_NB = {dS_dY_NB}")

# Second derivative  
d2S_dY_NB2 = dS_dY_NB.diff(Y_NB)
print(f"d²S/dY_NB² = {d2S_dY_NB2}")

# This should be just 8.3145*T/Y_NB
# Both CPU and GPU should get this same symbolic result

print("\n=== The Real Issue ===")
print("Both CPU and GPU compute the same symbolic derivatives.")
print("The difference must be in:")
print("1. How Piecewise expressions are handled")
print("2. How the C code is generated from the symbolic expressions")
print("3. Some post-processing or simplification step")