#!/usr/bin/env python3
"""Check what energy expression the model uses"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, Model
import pycalphad.variables as v

# Load database and create model
db = Database('NbTi.tdb')
model = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("Model state variables:", model.state_variables)
print("\nModel G expression (first 500 chars):")
print(str(model.G)[:500])
print("\nModel GM expression (first 500 chars):")
print(str(model.GM)[:500])

# Check the ratio
print("\nChecking if GM = G / (sum of site fractions):")
print("Site fractions:", model.site_fractions)
print("Sum of site fractions:", sum(model.site_fractions))

# Evaluate at test conditions
from sympy import symbols
T = symbols('T')
test_vals = {T: 1000}
for i, sf in enumerate(model.site_fractions):
    test_vals[sf] = 0.5  # Equal site fractions

try:
    G_val = float(model.G.subs(test_vals))
    GM_val = float(model.GM.subs(test_vals))
    print(f"\nAt T=1000K, equal site fractions:")
    print(f"G = {G_val}")
    print(f"GM = {GM_val}")
    print(f"Ratio GM/G = {GM_val/G_val}")
except Exception as e:
    print(f"Error evaluating: {e}")