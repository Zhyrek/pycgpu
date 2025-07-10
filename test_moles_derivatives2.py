#!/usr/bin/env python3
"""Test the moles derivatives - simplified"""
from pycalphad import Database, Model
import symengine as se

# Load database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Analyzing GPU vs CPU Difference ===")

# The key issue: 
# GPU is calculating d(moles_TI)/d(Y_NB) = -1
# CPU is calculating d(moles_TI)/d(Y_NB) = 0

print("\nFor BCC_A2 phase in NB-TI system:")
print("- moles(NB) = Y(BCC_A2,0,NB)")
print("- moles(TI) = Y(BCC_A2,0,TI)")

print("\nIf site fractions are independent:")
print("- d(moles_TI)/d(Y_NB) = 0  (CPU behavior)")

print("\nIf Y_TI = 1 - Y_NB (dependent):")
print("- moles(TI) = 1 - Y_NB")
print("- d(moles_TI)/d(Y_NB) = -1  (GPU behavior)")

print("\n=== The Issue ===")
print("The GPU code generation is treating Y_TI as dependent on Y_NB")
print("This happens because in a binary substitutional phase,")
print("the site fraction constraint Y_NB + Y_TI = 1 is being applied")
print("during differentiation, making one variable dependent.")

print("\n=== Checking Model Details ===")
# Check constituents
print(f"Phase constituents: {mod.constituents}")
print(f"Nonvacant elements: {mod.nonvacant_elements}")

# Check if VA is actually present
has_vacancies = any('VA' in str(species) for sublattice in mod.constituents for species in sublattice)
print(f"Has vacancies: {has_vacancies}")

if not has_vacancies:
    print("\nSince there are no vacancies in BCC_A2, we have:")
    print("- Only NB and TI on the sublattice")
    print("- Constraint: Y_NB + Y_TI = 1")
    print("- The model treats one as dependent")

print("\n=== Solution ===")
print("The GPU formulamole_grad function needs to be generated")
print("with independent site fractions, matching CPU behavior.")