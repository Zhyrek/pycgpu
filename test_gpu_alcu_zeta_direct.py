#!/usr/bin/env python
"""Test GPU equilibrium for ALCU_ZETA directly"""

from pycalphad import Database
from pycalphad.gpu.gpu_equilibrium import equilibrium_gpu
import pycalphad.variables as v

db = Database('Al-Cu-Fe.tdb')

conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.45,
    v.X('CU'): 0.55,
    v.N: 1
}

print("\nTesting ALCU_ZETA phase directly with GPU:")

try:
    # Test GPU equilibrium directly
    result = equilibrium_gpu(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                           calc_opts={'pdens': 50}, verbose=True)
    print(f"SUCCESS! GM = {result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()