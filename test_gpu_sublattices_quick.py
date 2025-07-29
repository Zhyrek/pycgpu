#!/usr/bin/env python
"""Quick test of GPU with different sublattice structures."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')
import os

# Disable CPU debug output
os.environ['PYCALPHAD_CPU_DEBUG'] = '0'

db = Database('Al-Cu-Fe.tdb')

conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.45,
    v.X('CU'): 0.55,
    v.N: 1
}

print("GPU Compilation Test Results:")
print("="*60)

# Test 1: LIQUID (1 sublattice, no VA)
print("1. LIQUID (1 sublattice, no VA): ", end='', flush=True)
try:
    equilibrium(db, ['AL','CU','FE','VA'], ['LIQUID'], conditions,
                calc_opts={'pdens': 10}, gpu=True, verbose=False)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED ({type(e).__name__})")

# Test 2: ALCU_ZETA (2 sublattices, no VA)  
print("2. ALCU_ZETA (2 sublattices, no VA): ", end='', flush=True)
try:
    equilibrium(db, ['AL','CU','FE','VA'], ['ALCU_ZETA'], conditions,
                calc_opts={'pdens': 10}, gpu=True, verbose=False)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED ({type(e).__name__})")

# Test 3: FCC_A1 (2 sublattices, with VA)
print("3. FCC_A1 (2 sublattices, with VA): ", end='', flush=True)
try:
    equilibrium(db, ['AL','CU','FE','VA'], ['FCC_A1'], conditions,
                calc_opts={'pdens': 10}, gpu=True, verbose=False)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED ({type(e).__name__})")

print("\nConclusion:")
print("- If ALCU_ZETA fails: Issue is with multiple sublattices")
print("- If ALCU_ZETA succeeds but FCC_A1 fails: Issue is specifically with VA")