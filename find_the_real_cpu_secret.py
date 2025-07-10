#!/usr/bin/env python3
"""Find the exact line where CPU avoids spurious terms"""

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

print("=== The Key Discovery ===")

# Get the exact symbolic hessian the CPU computes
variables = prf.state_variables + model.site_fractions
wrt = sympify(tuple(variables))
graph = sympify(model.G)

# Get hessian element [3,3]
grad_Y_NB = graph.diff(wrt[3])
hess_33_symbolic = grad_Y_NB.diff(wrt[3])

# Check if the symbolic hessian contains spurious terms
hess_str = str(hess_33_symbolic)
if 'BCC_A20TI**(-1)' in hess_str:
    print("FOUND: The symbolic hessian DOES contain Y_TI**(-1) terms!")
    
    # Count occurrences
    count_nb_inv = hess_str.count('BCC_A20NB**(-1')
    count_ti_inv = hess_str.count('BCC_A20TI**(-1')
    print(f"Y_NB**(-1) appears: {count_nb_inv} times")
    print(f"Y_TI**(-1) appears: {count_ti_inv} times")
    
    # Look for the specific pattern
    import re
    # Pattern: RT*(1/Y_NB + 1/Y_TI)
    pattern = r'8\.3145\*T\*\(.*?BCC_A20NB\*\*\(-1.*?\+.*?BCC_A20TI\*\*\(-1.*?\)'
    matches = re.findall(pattern, hess_str)
    if matches:
        print(f"\nFound {len(matches)} occurrences of RT*(1/Y_NB + 1/Y_TI) pattern")
        for m in matches[:2]:  # Show first 2
            print(f"  {m[:100]}...")

print("\n=== Now check build_functions ===")
# This is what PhaseRecordFactory uses
phase_rec = prf.get('BCC_A2')

# The secret might be in _get_lambidfy_options
try:
    from pycalphad.codegen.sympydiff_utils import _get_lambidfy_options
    opts = _get_lambidfy_options({})
    print(f"\nLambdify options: {opts}")
except:
    print("\nCouldn't get lambdify options")

# Check if there's any simplification happening
print("\n=== Testing Simplification ===")

# Create a test expression with the problematic structure
Y_NB = se.Symbol('BCC_A20NB')
Y_TI = se.Symbol('BCC_A20TI')
T = se.Symbol('T')

# This is the spurious term that appears in GPU
spurious = 8.3145 * T * (Y_NB**(-1) + Y_TI**(-1))
print(f"\nSpurious term: {spurious}")

# What happens when we lambdify it?
inp = sympify([v.N, v.P, T, Y_NB, Y_TI])
spurious_func = lambdify(inp, [spurious])

# Test it
test_vals = [1.0, 101325.0, 1000.0, 0.6, 0.4]
result = spurious_func(test_vals)
print(f"Spurious term evaluated: {result}")

# Now check what happens with the full hessian
print("\n=== The Answer ===")
print("CPU's symbolic hessian DOES contain the spurious Y_TI**(-1) terms!")
print("So the difference must be elsewhere...")

# The real secret might be in how phase_record_factory builds the functions
print("\n=== Checking PhaseRecordFactory ===")

# Get the actual hessian function
hess_func = phase_rec.formulahess

# Test it
dof = np.array([1.0, 101325.0, 1000.0, 0.6, 0.4])
hess = np.zeros((5, 5), order='C')
phase_rec.formulahess(hess, dof)

print(f"\nCPU hessian[3,3] = {hess[3,3]:.2f}")
print(f"Expected (RT/Y_NB): {8.3145 * 1000 / 0.6:.2f}")

# The ratio tells us if spurious terms are present
ratio = hess[3,3] / (8.3145 * 1000 / 0.6)
print(f"Ratio: {ratio:.3f}")

if abs(ratio - 1.0) < 0.01:
    print("\nCPU does NOT have spurious terms in the final result!")
    print("Despite having them in the symbolic expression!")