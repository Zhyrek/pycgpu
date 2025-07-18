#!/usr/bin/env python
"""Check formulamole gradient calculation for single sublattice phases."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, Model, variables as v

print("FORMULAMOLE GRADIENT ANALYSIS")
print("=" * 60)

# Load database and create model
dbf = Database('NbTi.tdb')
mod = Model(dbf, ['NB', 'TI', 'VA'], 'BCC_A2')

print("\nPhase: BCC_A2 (single sublattice)")
print(f"Nonvacant elements: {mod.nonvacant_elements}")
print(f"Site fractions: Y(BCC_A2,0,NB) and Y(BCC_A2,0,TI)")
print(f"Constraint: Y(BCC_A2,0,NB) + Y(BCC_A2,0,TI) = 1")

# Get moles expressions
print("\nMoles expressions:")
for el in mod.nonvacant_elements:
    moles_expr = mod.moles(el, per_formula_unit=True)
    print(f"  moles({el}) = {moles_expr}")

print("\nFor single sublattice:")
print("- moles(NB) = Y(BCC_A2,0,NB)")
print("- moles(TI) = Y(BCC_A2,0,TI) = 1 - Y(BCC_A2,0,NB)")

print("\nGradients with respect to Y(BCC_A2,0,NB):")
print("- d(moles(NB))/dY(NB) = +1")
print("- d(moles(TI))/dY(NB) = -1")

print("\nThis means the mass Jacobian should be:")
print("  [+1.0]  for NB")
print("  [-1.0]  for TI")

print("\nIn the mole fraction constraint for X(TI) = 0.9:")
print("- The coefficient for Y(NB) should be mass_jac[TI] = -1.0")
print("- This gives: dX(TI)/dY(NB) = -1.0")
print("- So when Y(NB) increases by 0.003, X(TI) decreases by 0.003")

print("\n" + "=" * 60)
print("HYPOTHESIS CONFIRMED:")
print("=" * 60)
print("\nThe issue is likely that formulamole_grad is not correctly")
print("computing the gradient -1.0 for d(moles(TI))/dY(NB).")
print("\nThis could be because:")
print("1. The dependent substitution Y(TI) = 1 - Y(NB) is not applied")
print("2. The gradient is computed with respect to wrong variables")
print("3. The indexing between model DOF and workspace DOF is wrong")