#!/usr/bin/env python
"""Debug constraint RHS initialization issue."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("Testing GPU constraint RHS initialization...")
print("Expected behavior: RHS should be set to target value (0.005)")
print("Current behavior: RHS is initialized to 0, residual subtracted from 0")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test with constraint X(TI) = 0.005
conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

print("\nRunning GPU equilibrium calculation with verbose output...")
print("Look for '[GPU MOLE FRAC]' lines to see RHS before/after residual")

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_final = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"\nFinal result:")
print(f"GPU X(TI) = {x_ti_final:.12f}")
print(f"Target    = 0.005000000000")
print(f"Error     = {x_ti_final - 0.005:.2e}")

print("\nDiagnosis:")
print("If RHS is always 0.0 in debug output, then the target value is missing!")
print("The constraint should read: current_X - target = 0")
print("But it's being set up as: current_X - 0 = residual")