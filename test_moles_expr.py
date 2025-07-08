#!/usr/bin/env python3
"""Test what moles expressions are generated"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phase = 'BCC_A2'

# Create model
model = Model(db, comps, phase)

print(f"Phase: {phase}")
print(f"Site fractions: {model.site_fractions}")
print(f"Nonvacant elements: {model.nonvacant_elements}")

for el in model.nonvacant_elements:
    moles_expr = model.moles(el, per_formula_unit=True)
    print(f"\nmoles({el}) = {moles_expr}")
    
    # Get the gradient symbolically
    import symengine
    grad = []
    for var in model.site_fractions:
        grad.append(symengine.diff(moles_expr, var))
    print(f"gradient with respect to {model.site_fractions}:")
    for i, g in enumerate(grad):
        print(f"  d(moles_{el})/d({model.site_fractions[i]}) = {g}")
    
    # Check what BCC_A20TI is
    if el == 'TI':
        print(f"\nChecking what BCC_A20TI variable is:")
        for var in model.ast.free_symbols:
            if 'BCC_A20TI' in str(var):
                print(f"  Found: {var}")
        # Check if it's equal to Y_TI
        print(f"  Y_TI = {model.site_fractions[1]}")