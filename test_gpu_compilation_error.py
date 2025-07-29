#!/usr/bin/env python
"""Capture GPU compilation error details for Al-Cu-Fe."""

from pycalphad import Database, equilibrium
import pycalphad.variables as v
import warnings
warnings.filterwarnings('ignore')

db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']
phases = list(db.phases.keys())

print("Testing GPU compilation for Al-Cu-Fe...")
print("="*60)

# Simple single-point test
conditions = {
    v.T: 1273.15,
    v.P: 101325,
    v.X('AL'): 0.3,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Attempting GPU calculation with verbose=True to see compilation errors...")
print()

try:
    # Run with verbose=True to see the compilation error
    result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, gpu=True, verbose=True)
    print("Unexpected success!")
except Exception as e:
    print(f"\nGPU compilation failed as expected.")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)[:200]}...")
    
    # Try with just one phase to narrow down the issue
    print("\n" + "="*60)
    print("Testing individual phases to find which one fails compilation:")
    
    for phase in ['LIQUID', 'FCC_A1', 'BCC_A2', 'ALCU_ZETA', 'AL2FE']:
        print(f"\n{phase}: ", end='', flush=True)
        try:
            result = equilibrium(db, components, [phase], conditions,
                               calc_opts={'pdens': 10}, gpu=True, verbose=False)
            print("SUCCESS")
        except Exception as e:
            print(f"FAILED - {type(e).__name__}")