#!/usr/bin/env python3
"""Debug the moles expression generation in GPU code"""
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
import symengine as se

# Load database and create model
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create workspace
components = ['NB', 'TI']
wks = Workspace(db, components, phases, {})

# Get the model
model = wks.models['BCC_A2']

print("=== Debugging Moles Expression Generation ===")
print(f"Phase: {model.phase_name}")
print(f"Nonvacant elements: {model.nonvacant_elements}")
print(f"Site fractions: {model.site_fractions}")

# Get moles expressions
print("\nMoles expressions:")
for el in model.nonvacant_elements:
    moles_expr = model.moles(el, per_formula_unit=True)
    print(f"\nmoles({el}) = {moles_expr}")
    
    # Get the derivatives
    print(f"Derivatives of moles({el}):")
    for sf in model.site_fractions:
        deriv = se.diff(moles_expr, sf)
        print(f"  d(moles_{el})/d({sf}) = {deriv}")

# Check what variables are used
print("\n=== Variable Analysis ===")
all_vars = set()
for el in model.nonvacant_elements:
    moles_expr = model.moles(el, per_formula_unit=True)
    vars_in_expr = moles_expr.free_symbols
    all_vars.update(vars_in_expr)
    print(f"Variables in moles({el}): {vars_in_expr}")

print(f"\nAll variables used: {all_vars}")

# Check if any constraints are being applied
print("\n=== Checking Constraints ===")
print(f"Internal constraints: {model.get_internal_constraints()}")

# Check the site fraction mapping
print("\n=== Site Fraction Mapping ===")
print("AST to pycalphad variable mapping:")
for ast_var in model.ast.atoms(se.Symbol):
    if str(ast_var).startswith('BCC_A2'):
        print(f"  {ast_var}")