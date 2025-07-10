#!/usr/bin/env python3
"""Test the moles derivatives in detail"""
from pycalphad import Database, Model
import symengine as se
import numpy as np

# Load database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Testing Moles Derivatives ===")

# Get site fractions as symbols
y_nb = None
y_ti = None
for sf in mod.site_fractions:
    if 'NB' in str(sf):
        y_nb = sf
    elif 'TI' in str(sf):
        y_ti = sf

print(f"Site fractions: Y_NB = {y_nb}, Y_TI = {y_ti}")

# Get moles expressions
moles_nb_expr = mod.moles('NB', per_formula_unit=True)
moles_ti_expr = mod.moles('TI', per_formula_unit=True)

print(f"\nmoles(NB) = {moles_nb_expr}")
print(f"moles(TI) = {moles_ti_expr}")

# The expressions use BCC_A20NB and BCC_A20TI
# Let's see what happens if we substitute the actual site fractions

# Get the AST symbols
bcc_nb = se.symbols('BCC_A20NB')
bcc_ti = se.symbols('BCC_A20TI')

print(f"\n=== Direct Substitution Test ===")
# If we substitute BCC_A20NB -> y_nb and BCC_A20TI -> y_ti
moles_nb_direct = moles_nb_expr.subs(bcc_nb, y_nb).subs(bcc_ti, y_ti)
moles_ti_direct = moles_ti_expr.subs(bcc_nb, y_nb).subs(bcc_ti, y_ti)

print(f"moles(NB) with substitution = {moles_nb_direct}")
print(f"moles(TI) with substitution = {moles_ti_direct}")

# Calculate derivatives
print(f"\n=== Derivatives (treating as independent) ===")
d_moles_nb_d_ynb = se.diff(moles_nb_direct, y_nb)
d_moles_nb_d_yti = se.diff(moles_nb_direct, y_ti)
d_moles_ti_d_ynb = se.diff(moles_ti_direct, y_nb)
d_moles_ti_d_yti = se.diff(moles_ti_direct, y_ti)

print(f"d(moles_NB)/d(Y_NB) = {d_moles_nb_d_ynb}")
print(f"d(moles_NB)/d(Y_TI) = {d_moles_nb_d_yti}")
print(f"d(moles_TI)/d(Y_NB) = {d_moles_ti_d_ynb}")
print(f"d(moles_TI)/d(Y_TI) = {d_moles_ti_d_yti}")

# Now test with constraint Y_TI = 1 - Y_NB
print(f"\n=== Derivatives with constraint Y_TI = 1 - Y_NB ===")
# Substitute Y_TI with (1 - Y_NB)
moles_nb_constrained = moles_nb_direct.subs(y_ti, 1 - y_nb)
moles_ti_constrained = moles_ti_direct.subs(y_ti, 1 - y_nb)

print(f"moles(NB) with constraint = {moles_nb_constrained}")
print(f"moles(TI) with constraint = {moles_ti_constrained}")

d_moles_nb_d_ynb_c = se.diff(moles_nb_constrained, y_nb)
d_moles_ti_d_ynb_c = se.diff(moles_ti_constrained, y_nb)

print(f"d(moles_NB)/d(Y_NB) with constraint = {d_moles_nb_d_ynb_c}")
print(f"d(moles_TI)/d(Y_NB) with constraint = {d_moles_ti_d_ynb_c}")

print("\nThis shows that if Y_TI is treated as dependent (Y_TI = 1 - Y_NB),")
print("then d(moles_TI)/d(Y_NB) = -1, which is what the GPU is calculating!")

# The issue is that the GPU code generation is using dependent site fractions