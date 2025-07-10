#!/usr/bin/env python3
"""Debug the variable conversion in GPU code generation"""
from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
from pycalphad.gpu.gpu_codegen import notebook_convert_var_names
import symengine as se

# Load database and create model
db = Database('NbTi.tdb')
components = ['NB', 'TI']
phases = ['BCC_A2']

# Create workspace
wks = Workspace(db, components, phases, {})

# Get the model
model = wks.models['BCC_A2']

print("=== Debugging Variable Conversion ===")

# Test expressions
test_expressions = [
    "1.0*BCC_A20NB",
    "1.0*BCC_A20TI",
    "1 - BCC_A20NB",  # If TI is being treated as dependent
    "BCC_A20NB + BCC_A20TI"
]

print("\nTesting variable conversion:")
for expr_str in test_expressions:
    converted = notebook_convert_var_names(expr_str, model, wks)
    print(f"{expr_str} -> {converted}")

# Check the actual moles expressions
print("\n=== Actual Moles Expressions ===")
for el in model.nonvacant_elements:
    moles_expr = model.moles(el, per_formula_unit=True)
    expr_str = str(moles_expr)
    converted = notebook_convert_var_names(expr_str, model, wks)
    print(f"moles({el}): {expr_str} -> {converted}")

# Check derivatives
print("\n=== Checking Derivatives ===")
moles_ti = model.moles('TI', per_formula_unit=True)
y_nb = model.site_fractions[0]  # Y(BCC_A2,0,NB)

# Manual derivative
deriv = se.diff(moles_ti, y_nb)
print(f"d(moles_TI)/d(Y_NB) = {deriv}")

# Convert to string and check conversion
deriv_str = str(deriv)
converted_deriv = notebook_convert_var_names(deriv_str, model, wks)
print(f"Converted derivative: {deriv_str} -> {converted_deriv}")

# Check if there's any substitution happening
print("\n=== Checking for Dependent Variable Substitution ===")
# If TI is treated as dependent, we'd see Y_TI = 1 - Y_NB
print("Internal constraints:", model.get_internal_constraints())
constraint_str = str(model.get_internal_constraints()[0])
converted_constraint = notebook_convert_var_names(constraint_str, model, wks)
print(f"Constraint: {constraint_str} -> {converted_constraint}")