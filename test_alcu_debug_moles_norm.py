#!/usr/bin/env python
"""Test to debug moles_normalization values for multi-sublattice phases."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Testing moles_normalization for multi-sublattice phase")
print("="*70)
print("ALCU_ZETA has site ratios (9.0, 11.0), sum = 20.0")
print("LIQUID has site ratio 1.0")
print("="*70)

# Test one condition
conditions = {
    v.T: 900, 
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

print("\nRunning equilibrium calculation...")
print("Conditions: T=900K, X(AL)=0.6, X(CU)=0.3, X(FE)=0.1")

# Run calculation
try:
    result = equilibrium(dbf, comps, phases, conditions, 
                        calc_opts={'pdens': 100}, verbose=False, gpu=True)
    
    print("\nKey values to look for in debug output:")
    print("1. moles_normalization values for each phase")
    print("2. Site fractions and how they relate to mole fractions")
    print("3. Matrix coefficients in system amount constraint row")
    print("4. Phase amounts (formula units) vs mole fractions")
    
except Exception as e:
    print(f"Error: {e}")