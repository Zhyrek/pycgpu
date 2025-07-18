#!/usr/bin/env python
"""Test to understand delta_y calculation after consolidation."""

# From the test output after consolidation (iteration 2):
c_G = [2.214892280104281e-01, -2.214892280104288e-01]
c_component = [[3.506507e-05, -3.506507e-05], 
                [-3.506507e-05, 3.506507e-05]]
chemical_potentials = [-25739.210873, -19284.602909]

print("Manual delta_y calculation:")
print("c_G[0] =", c_G[0])
print("Chemical potential contribution:")
delta_y_0_chem = c_component[0][0] * chemical_potentials[0] + c_component[1][0] * chemical_potentials[1]
print(f"  c_component[0][0] * mu[0] = {c_component[0][0]} * {chemical_potentials[0]} = {c_component[0][0] * chemical_potentials[0]}")
print(f"  c_component[1][0] * mu[1] = {c_component[1][0]} * {chemical_potentials[1]} = {c_component[1][0] * chemical_potentials[1]}")
print(f"  Total chemical potential contribution = {delta_y_0_chem}")

delta_y_0 = c_G[0] + delta_y_0_chem
print(f"\ndelta_y[0] = c_G[0] + chem_contrib = {c_G[0]} + {delta_y_0_chem} = {delta_y_0}")

print("\nBut GPU reports delta_y[0] = 2.865204599913607e-06")
print("Difference:", delta_y_0 - 2.865204599913607e-06)