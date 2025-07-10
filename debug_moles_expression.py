#!/usr/bin/env python3
"""Debug the actual moles expressions used by the model"""
from pycalphad import Database, Model
import symengine as se

# Load the database and create model
db = Database('NbTi.tdb')
mod = Model(db, ['NB', 'TI', 'VA'], 'BCC_A2')

print("=== BCC_A2 Moles Expressions Debug ===")
print(f"Phase: {mod.phase_name}")
print(f"Constituents: {mod.constituents}")
print(f"Site fractions: {mod.site_fractions}")

# Get the actual SymEngine symbols
y_nb = None
y_ti = None
for sf in mod.site_fractions:
    if 'NB' in str(sf):
        y_nb = sf
    elif 'TI' in str(sf):
        y_ti = sf

print(f"\nSymbol for Y_NB: {y_nb}")
print(f"Symbol for Y_TI: {y_ti}")

# Check what BCC_A20NB and BCC_A20TI are
print("\n=== Checking BCC_A20NB and BCC_A20TI symbols ===")
# These are internal symbols used in the moles expressions
# Let's see what they represent

moles_nb = mod.moles('NB', per_formula_unit=True)
moles_ti = mod.moles('TI', per_formula_unit=True)

print(f"\nmoles(NB) = {moles_nb}")
print(f"moles(TI) = {moles_ti}")

# Extract all symbols from these expressions
symbols_in_nb = list(moles_nb.free_symbols)
symbols_in_ti = list(moles_ti.free_symbols)

print(f"\nSymbols in moles(NB): {symbols_in_nb}")
print(f"Symbols in moles(TI): {symbols_in_ti}")

# Now let's manually check what BCC_A20NB and BCC_A20TI are
# They should be site fractions, but let's verify

# Check the ast_ (Abstract Syntax Tree) dictionary
print("\n=== Checking AST dictionary ===")
if hasattr(mod, 'ast_'):
    for key, value in mod.ast_.items():
        if 'BCC_A20' in str(key):
            print(f"{key}: {value}")

# Check if there's any substitution happening
print("\n=== Checking for dependent site fractions ===")
# In a binary substitutional phase, one site fraction might be dependent
# Let's see if the model is using Y_TI = 1 - Y_NB

# First, let's check the site fraction constraint
total_sf = sum(mod.site_fractions)
print(f"Sum of site fractions: {total_sf}")

# Check if there's any constraint being applied
if hasattr(mod, '_site_fraction_constraints'):
    print(f"Site fraction constraints: {mod._site_fraction_constraints}")

# Let's also check the phase constituents more carefully
print("\n=== Phase constituents details ===")
for i, sublattice in enumerate(mod.constituents):
    print(f"Sublattice {i}: {sublattice}")
    
# The key insight: BCC_A20NB and BCC_A20TI might be referring to 
# internal variables that are substituted later