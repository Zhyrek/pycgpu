#!/usr/bin/env python
"""Check if GPU works with LIQUID and ALCU_ZETA phases."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')

# Suppress debug output
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Load database
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Simple conditions
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing phases with GPU (this may take a moment)...")
print("="*60)

# Test individual phases
for phase in ['LIQUID', 'ALCU_ZETA']:
    print(f"\nTesting {phase} only...", end='', flush=True)
    try:
        result = equilibrium(db, components, [phase], conditions, 
                           calc_opts={'pdens': 2}, 
                           gpu=True, 
                           verbose=False)
        print(f" ✓ SUCCESS")
    except Exception as e:
        print(f" ✗ FAILED ({type(e).__name__})")

# Test both phases together
print(f"\nTesting LIQUID + ALCU_ZETA...", end='', flush=True)
try:
    result = equilibrium(db, components, ['LIQUID', 'ALCU_ZETA'], conditions, 
                       calc_opts={'pdens': 2}, 
                       gpu=True, 
                       verbose=False)
    print(f" ✓ SUCCESS")
    print(f"\nGPU calculation completed!")
    print(f"  GM = {result.GM.values[0]:.1f} J/mol")
    for phase in ['LIQUID', 'ALCU_ZETA']:
        np_val = result.NP.sel(phase=phase).values[0]
        if np_val > 1e-10:
            print(f"  {phase}: {np_val:.4f}")
except Exception as e:
    print(f" ✗ FAILED")
    print(f"  Error: {type(e).__name__}")
    if 'nvcc' in str(e).lower():
        print("  This is an nvcc compilation error")
        print("  The generated code may be too complex")
    else:
        print(f"  Message: {str(e)[:200]}")