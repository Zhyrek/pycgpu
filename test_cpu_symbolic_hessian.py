#!/usr/bin/env python3
"""Test CPU symbolic hessian computation to understand how G is differentiated"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import symengine as se
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']

# Build model for BCC_A2
mod = Model(db, comps, 'BCC_A2')

print("Model components:", mod.components)
print("Model constituents:", mod.constituents)
print("Site ratios:", mod.site_ratios)
print()

# Get the G expression (formula energy)
G_expr = mod.G
print("G expression (formula energy):")
print(G_expr)
print()

# Get GM expression (molar energy)
GM_expr = mod.GM
print("GM expression (molar energy):")
print(GM_expr)
print()

# Get site ratio normalization
site_ratio_norm = mod._site_ratio_normalization
print("Site ratio normalization:")
print(site_ratio_norm)
print()

# Verify that G = GM * site_ratio_normalization
print("Verifying G = GM * site_ratio_normalization:")
diff = G_expr - GM_expr * site_ratio_norm
print("G - GM * site_ratio_norm =", diff)
print()

# Get the site fraction variables
site_fracs = [v for v in mod.variables if hasattr(v, 'sublattice_index')]
print("Site fraction variables:", site_fracs)
print()

# Compute the Hessian of G with respect to site fractions
print("Computing Hessian of G with respect to site fractions:")
for i, sf1 in enumerate(site_fracs):
    for j, sf2 in enumerate(site_fracs):
        h_ij = G_expr.diff(sf1).diff(sf2)
        print(f"d²G/d{sf1}d{sf2} = {h_ij}")
print()

# Now let's evaluate at a specific point
T = 1000.0
P = 101325.0
Y_NB = 0.612245
Y_TI = 0.387755

# Create substitution dictionary
subs = {
    'T': T,
    'P': P,
    'Y(BCC_A2,0,NB)': Y_NB,
    'Y(BCC_A2,0,TI)': Y_TI
}

# Convert string keys to symbols
import pycalphad.variables as v
subs_symb = {}
for key, val in subs.items():
    if key == 'T':
        subs_symb[v.T] = val
    elif key == 'P':
        subs_symb[v.P] = val
    elif 'Y(' in key:
        # Parse site fraction
        parts = key.replace('Y(', '').replace(')', '').split(',')
        phase = parts[0]
        subl_idx = int(parts[1])
        species = parts[2]
        subs_symb[v.SiteFraction(phase, subl_idx, species)] = val

print("Evaluating Hessian at specific values:")
print(f"T = {T}, P = {P}, Y_NB = {Y_NB}, Y_TI = {Y_TI}")
print()

# Compute numerical values of Hessian elements
for i, sf1 in enumerate(site_fracs):
    for j, sf2 in enumerate(site_fracs):
        h_ij = G_expr.diff(sf1).diff(sf2)
        h_ij_val = float(h_ij.xreplace(subs_symb))
        print(f"H[{i},{j}] = {h_ij_val:.6e}")