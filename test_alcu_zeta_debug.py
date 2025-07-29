#!/usr/bin/env python
"""Debug test for ALCU_ZETA phase GPU compilation."""

from pycalphad import Database, equilibrium, Model
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')

db = Database('Al-Cu-Fe.tdb')

# First, check the model structure
mod = Model(db, ['AL', 'CU', 'FE'], 'ALCU_ZETA')
print("ALCU_ZETA Model Analysis:")
print(f"  Number of site fractions: {len(mod.site_fractions)}")
print(f"  Site fractions: {[str(sf) for sf in mod.site_fractions]}")
print(f"  Energy expression type: {type(mod.GM)}")
print(f"  Has Piecewise: {'Piecewise' in str(mod.GM)}")

# Now try GPU equilibrium with verbose output
print("\nTesting GPU equilibrium with verbose=True:")
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.45,
    v.X('CU'): 0.55,
    v.N: 1
}

try:
    result = equilibrium(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                        calc_opts={'pdens': 50}, gpu=True, verbose=True)
    print(f"SUCCESS! GM = {result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()