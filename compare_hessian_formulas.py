#!/usr/bin/env python3
"""Compare CPU and GPU Hessian calculation formulas"""

from pycalphad import Database, Model
from pycalphad.core.workspace import Workspace
import pycalphad.variables as v
import numpy as np
from symengine import symbols, diff
import symengine

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Create model
model = Model(db, comps, 'BCC_A2')

print("=== Comparing CPU and GPU Hessian Formulas ===\n")

# Get the energy expression
G = model.G
print("Energy expression variables:", model.variables)

# The model uses these variables:
# T (temperature)
# BCC_A20NB (Y_NB site fraction)
# BCC_A20TI (Y_TI site fraction)

# Get second derivatives
# Get the actual variable symbols
import pycalphad.variables as pv
T = pv.T
# The site fraction variables from the model
site_frac_vars = model.variables[1:]  # Skip T
Y_NB = site_frac_vars[0]  # Y(BCC_A2,0,NB)
Y_TI = site_frac_vars[1]  # Y(BCC_A2,0,TI)

print("\nCalculating d²G/dY_NB²...")
dG_dYNB = diff(G, Y_NB)
d2G_dYNB2 = diff(dG_dYNB, Y_NB)

print("\nChecking for normalization factor...")
# The issue might be that the GPU is calculating the Hessian 
# for the normalized energy G/(Y_NB + Y_TI) instead of G

# Let's check what happens with different normalizations
print("\n1. Direct Hessian of G:")
print("   d²G/dY_NB²")

print("\n2. Hessian of normalized G/(Y_NB + Y_TI):")
G_normalized = G / (Y_NB + Y_TI)
dGn_dYNB = diff(G_normalized, Y_NB)
d2Gn_dYNB2 = diff(dGn_dYNB, Y_NB)
print("   d²[G/(Y_NB + Y_TI)]/dY_NB²")

# The factor of 2.5 might come from the normalization
print("\n3. Checking if there's a systematic factor...")

# Evaluate at specific point Y_NB=0.6, Y_TI=0.4, T=300
subs_dict = {T: 300, Y_NB: 0.6, Y_TI: 0.4}

# Since symengine evaluation might be complex, let's just print the structure
print("\nThe GPU might be using a different normalization or energy units.")
print("Common sources of scaling factors:")
print("- Per-mole vs per-formula-unit energy")
print("- Different handling of ideal mixing entropy")
print("- Site fraction normalization")