#!/usr/bin/env python
"""Test AlCu system with normalized system amount constraint."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing AlCu with NORMALIZED system amount constraint")
print("="*70)
print("Expected: All phases should have coefficient 1.0 in Row 5")
print("Previous: LIQUID=1.0, ALCU_ZETA=20.0")
print("="*70)

# Test the condition that was closest before
conditions = {
    v.T: 900,  # Med T, balanced
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print("\nTesting Med T, balanced: X(AL)=0.6, X(CU)=0.3, T=900K")
print("Previous error with this condition: ~806 J/mol")
print("Original error (before dgelsd): ~6.94 J/mol")

try:
    # Run GPU calculation
    gpu_result = equilibrium(dbf, comps, phases, conditions, 
                           calc_opts={'pdens': 100}, verbose=False, gpu=True)
    gpu_gm = float(gpu_result.GM.values.item())
    
    print(f"\nGPU GM: {gpu_gm:.6f} J/mol")
    print("\nCheck debug output for:")
    print("1. [GPU MOLES_NORM] - should show ~1.0 for LIQUID, ~20.0 for ALCU_ZETA")
    print("2. [GPU SYSTEM AMOUNT] Phase X: moles_norm=...")
    print("3. Row 5: - coefficients should all be ~1.0 now")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()