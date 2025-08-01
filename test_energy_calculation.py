#!/usr/bin/env python
"""
Test energy calculation for AU2BI_C15 phase
"""

from pycalphad import Database, Model
import numpy as np

# Load database
db = Database('/mnt/c/users/scott/Documents/pycalphad/AuBi-07Wan.tdb')
components = ['AU', 'BI', 'VA']

# Create models
mod_c15 = Model(db, components, 'AU2BI_C15')
mod_liquid = Model(db, components, 'LIQUID')

# Test conditions
T = 400
P = 101325

# For AU2BI_C15, need to satisfy site ratios (2.0, 1.0)
# First sublattice: AU/BI, Second sublattice: AU/BI
# Let's use a simple case: all AU in first, all BI in second
y_c15 = [1.0, 0.0, 0.0, 1.0]  # [Y(AU2BI_C15,0,AU), Y(AU2BI_C15,0,BI), Y(AU2BI_C15,1,AU), Y(AU2BI_C15,1,BI)]

# For LIQUID: just AU and BI
y_liquid = [0.7, 0.3]  # [Y(LIQUID,0,AU), Y(LIQUID,0,BI)]

print("Testing energy calculations...")
print(f"T = {T} K, P = {P} Pa")

# Calculate energies
from symengine import symbols
from pycalphad import variables as pycalphad_v
N, P_var, T_var = pycalphad_v.N, pycalphad_v.P, pycalphad_v.T

# Build DOF for C15
dof_c15 = {N: 1.0, P_var: P, T_var: T}
# Add site fractions for C15
for var in mod_c15.variables:
    if 'Y(' in str(var):
        # Parse the variable name to get indices
        if 'AU2BI_C15,0,AU' in str(var):
            dof_c15[var] = y_c15[0]
        elif 'AU2BI_C15,0,BI' in str(var):
            dof_c15[var] = y_c15[1]
        elif 'AU2BI_C15,1,AU' in str(var):
            dof_c15[var] = y_c15[2]
        elif 'AU2BI_C15,1,BI' in str(var):
            dof_c15[var] = y_c15[3]

# Build DOF for LIQUID
dof_liquid = {N: 1.0, P_var: P, T_var: T}
for var in mod_liquid.variables:
    if 'Y(' in str(var):
        if 'LIQUID,0,AU' in str(var):
            dof_liquid[var] = y_liquid[0]
        elif 'LIQUID,0,BI' in str(var):
            dof_liquid[var] = y_liquid[1]

# Calculate G (per formula unit)
G_c15 = float(mod_c15.G.subs(dof_c15))
G_liquid = float(mod_liquid.G.subs(dof_liquid))

# Calculate GM (per mole of atoms)
GM_c15 = float(mod_c15.GM.subs(dof_c15))
GM_liquid = float(mod_liquid.GM.subs(dof_liquid))

print("\nAU2BI_C15:")
print(f"  Site fractions: {y_c15}")
print(f"  G (per formula unit): {G_c15:.6f}")
print(f"  GM (per mole atoms): {GM_c15:.6f}")
print(f"  G/GM ratio: {G_c15/GM_c15:.6f} (should be site ratio sum = 3.0)")

print("\nLIQUID:")
print(f"  Site fractions: {y_liquid}")
print(f"  G (per formula unit): {G_liquid:.6f}")
print(f"  GM (per mole atoms): {GM_liquid:.6f}")
print(f"  G/GM ratio: {G_liquid/GM_liquid:.6f} (should be site ratio sum = 1.0)")

# Check equilibrium result
phase_fractions = {'AU2BI_C15': 0.897846, 'LIQUID': 0.102154}
print("\nExpected GM calculation:")
print(f"  GM = {phase_fractions['AU2BI_C15']} * {GM_c15:.2f} + {phase_fractions['LIQUID']} * {GM_liquid:.2f}")
print(f"     = {phase_fractions['AU2BI_C15'] * GM_c15 + phase_fractions['LIQUID'] * GM_liquid:.6f}")

print("\nIf GPU uses G instead of GM:")
print(f"  GM = {phase_fractions['AU2BI_C15']} * {G_c15:.2f} + {phase_fractions['LIQUID']} * {G_liquid:.2f}")
print(f"     = {phase_fractions['AU2BI_C15'] * G_c15 + phase_fractions['LIQUID'] * G_liquid:.6f}")