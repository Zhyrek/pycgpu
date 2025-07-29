#!/usr/bin/env python
"""Test equilibrium calculation with VA in sublattices."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v
import traceback

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

print("Testing equilibrium with VA in sublattices")
print("="*60)

# Test FCC_A1 only (has VA in second sublattice)
print("\nFCC_A1 phase (AL,CU,FE):(VA):")

# CPU calculation
try:
    print("  CPU calculation...", end='', flush=True)
    cpu_result = equilibrium(db, components, ['FCC_A1'], conditions, calc_opts={'pdens': 50})
    print(f" Success! GM = {cpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f" Failed: {e}")

# GPU calculation with verbose output
try:
    print("  GPU calculation with verbose output:")
    gpu_result = equilibrium(db, components, ['FCC_A1'], conditions, calc_opts={'pdens': 50}, gpu=True, verbose=True)
    print(f"  Success! GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f"  Failed with error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Also test LIQUID which does NOT have VA in sublattices
print("\n" + "="*60)
print("LIQUID phase (AL,CU,FE) - no VA in sublattices:")

# GPU calculation
try:
    print("  GPU calculation...", end='', flush=True)
    gpu_result = equilibrium(db, components, ['LIQUID'], conditions, calc_opts={'pdens': 50}, gpu=True, verbose=False)
    print(f" Success! GM = {gpu_result.GM.values[0,0,0,0]:.1f} J/mol")
except Exception as e:
    print(f" Failed: {e}")