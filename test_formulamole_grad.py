#!/usr/bin/env python3
"""Test the formulamole gradient generation"""
from pycalphad import Database, Model
import numpy as np

# Load the database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Testing BCC_A2 moles expressions ===")
print(f"Phase: {mod.phase_name}")
print(f"Nonvacant elements: {mod.nonvacant_elements}")
print(f"Site fractions: {mod.site_fractions}")

# Get moles expressions
for el in mod.nonvacant_elements:
    moles_expr = mod.moles(el, per_formula_unit=True)
    print(f"\nmoles({el}) = {moles_expr}")
    
# Let's also check the variables
print(f"\nVariables: {mod.variables}")

# Check what Y_TI refers to in the model
import symengine
y_vars = [v for v in mod.variables if str(v).startswith('Y')]
print(f"\nSite fraction variables: {y_vars}")

# Manually check derivatives
print("\n=== Manual derivative check ===")
for el in mod.nonvacant_elements:
    moles_expr = mod.moles(el, per_formula_unit=True)
    print(f"\nmoles({el}) = {moles_expr}")
    for var in y_vars:
        deriv = symengine.diff(moles_expr, var)
        print(f"  d(moles_{el})/d({var}) = {deriv}")

# Check if there's a constraint
print("\n=== Checking for constraints ===")
# In a binary substitutional phase, we have Y_NB + Y_TI = 1
# So Y_TI might be replaced by (1 - Y_NB) internally

# Let's create test values
Y_NB = 0.6
Y_TI = 0.4

# Create the state array
# The model uses workspace coordinates [N, P, T, Y1, Y2, ...]
# For BCC_A2, we need to know which index corresponds to which element
print(f"\nDegrees of freedom: {mod.degrees_of_freedom}")

# Create a simple test 
print("\n=== Testing actual values ===")
# Workspace format: [N, P, T, site_fractions...]
dof = np.array([1.0, 101325.0, 1000.0, Y_NB, Y_TI])
print(f"DOF array: {dof}")

# The issue might be that the model is using dependent site fractions
# In a single sublattice model with 2 components, one site fraction is dependent
# Y_TI = 1 - Y_NB

# Check the constituents
print(f"\nConstituents: {mod.constituents}")
print(f"Components: {mod.components}")

# For BCC_A2, it's a single sublattice with NB, TI, VA
# The site fraction constraint is Y_NB + Y_TI + Y_VA = 1
# But since VA is vacancy and this is a substitutional phase,
# Y_VA = 0, so Y_NB + Y_TI = 1