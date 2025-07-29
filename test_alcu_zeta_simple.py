#!/usr/bin/env python
"""Simple test for ALCU_ZETA phase GPU compilation."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v

db = Database('Al-Cu-Fe.tdb')

conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.45,
    v.X('CU'): 0.55,
    v.N: 1
}

print("Testing ALCU_ZETA phase:")

# CPU test
print("  CPU: ", end='', flush=True)
try:
    cpu_result = equilibrium(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                           calc_opts={'pdens': 50})
    print(f"SUCCESS, GM = {cpu_result.GM.values[0,0,0,0]:.1f} J/mol")
    cpu_success = True
except Exception as e:
    print(f"FAILED - {type(e).__name__}")
    cpu_success = False

# GPU test
print("  GPU: ", end='', flush=True)
try:
    gpu_result = equilibrium(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                           calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"SUCCESS, GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
    if cpu_success:
        diff = abs(gpu_result.GM.values[0,0,0,0] - cpu_result.GM.values[0,0,0,0])
        print(f"  CPU-GPU difference: {diff:.3f} J/mol")
except Exception as e:
    print(f"FAILED - {type(e).__name__}: {e}")