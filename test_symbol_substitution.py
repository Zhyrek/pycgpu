#!/usr/bin/env python3
"""Test if there's any symbol substitution happening"""
from pycalphad import Database, Model
import symengine as se

# Load database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== Checking for Symbol Substitutions ===")

# Get the moles expressions
moles_nb = mod.moles('NB', per_formula_unit=True)
moles_ti = mod.moles('TI', per_formula_unit=True)

print(f"moles(NB) = {moles_nb}")
print(f"moles(TI) = {moles_ti}")

# Check what BCC_A20NB and BCC_A20TI actually are
print("\n=== Symbol Details ===")

# Get all free symbols
all_symbols = set()
all_symbols.update(moles_nb.free_symbols)
all_symbols.update(moles_ti.free_symbols)

for sym in all_symbols:
    print(f"Symbol: {sym}, type: {type(sym)}")

# Check if there are any substitution dictionaries
print("\n=== Checking Model Attributes ===")
if hasattr(mod, '_symbols'):
    print(f"mod._symbols: {mod._symbols}")
    
if hasattr(mod, 'ast_'):
    print("\nmod.ast_ contents:")
    for key, value in mod.ast_.items():
        if 'BCC' in str(key) or 'Y(' in str(key):
            print(f"  {key}: {value}")

# Let's also check the energy expression to see if there's a constraint
energy = mod.GM
print(f"\n=== Energy Expression ===")
print(f"Has Y(BCC_A2,0,NB): {'Y(BCC_A2,0,NB)' in str(energy)}")
print(f"Has Y(BCC_A2,0,TI): {'Y(BCC_A2,0,TI)' in str(energy)}")
print(f"Has BCC_A20NB: {'BCC_A20NB' in str(energy)}")
print(f"Has BCC_A20TI: {'BCC_A20TI' in str(energy)}")

# Check if there's a site fraction constraint being applied
print("\n=== Site Fraction Analysis ===")
y_nb = None
y_ti = None
for sf in mod.site_fractions:
    if 'NB' in str(sf):
        y_nb = sf
    elif 'TI' in str(sf):
        y_ti = sf
        
print(f"Y_NB symbol: {y_nb}")
print(f"Y_TI symbol: {y_ti}")

# Check if the sum equals 1
if y_nb and y_ti:
    # In a binary substitutional model, Y_NB + Y_TI = 1
    # So Y_TI = 1 - Y_NB, which gives d(Y_TI)/d(Y_NB) = -1
    print("\nIn a binary substitutional model:")
    print("Y_NB + Y_TI = 1")
    print("Therefore: Y_TI = 1 - Y_NB")
    print("This gives: d(Y_TI)/d(Y_NB) = -1")
    print("\nThis explains why d(moles_TI)/d(Y_NB) = -1 in the GPU!")

# The issue is that the GPU code is using a dependent site fraction