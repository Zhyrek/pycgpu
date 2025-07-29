#!/usr/bin/env python
"""Test GPU equilibrium with VA in sublattices."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import traceback
import os

# Suppress debug output from CPU
os.environ['PYCALPHAD_DEBUG'] = '0'

# Test with Al-Cu-Fe which has VA in sublattices
db = Database('Al-Cu-Fe.tdb')
components = ['AL', 'CU', 'FE', 'VA']

# Simple test condition
conditions = {
    v.T: 1000,
    v.P: 101325,
    v.X('AL'): 0.5,
    v.X('CU'): 0.3,
    v.N: 1
}

print("Testing GPU with VA in sublattices")
print("="*60)

# Test FCC_A1 only (has VA in second sublattice)
print("\nFCC_A1 phase (AL,CU,FE):(VA):")
try:
    print("  GPU calculation...")
    gpu_result = equilibrium(db, components, ['FCC_A1'], conditions, 
                           calc_opts={'pdens': 50}, gpu=True, verbose=True)
    print(f"  Success! GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"  Failed with error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
print("\n" + "="*60)

# Test LIQUID which does NOT have VA in sublattices  
print("\nLIQUID phase (AL,CU,FE) - no VA in sublattices:")
try:
    print("  GPU calculation...")
    gpu_result = equilibrium(db, components, ['LIQUID'], conditions, 
                           calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f"  Success! GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"  Failed with error: {e}")
    traceback.print_exc()