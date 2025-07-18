#!/usr/bin/env python
"""Simple GPU constraint check - no debug output."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the constraint X(TI) = 0.005
conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_final = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"GPU X(TI) = {x_ti_final:.12f}")
print(f"Target    = 0.005000000000")
print(f"Error     = {abs(x_ti_final - 0.005):.2e}")

if abs(x_ti_final - 0.005) < 1e-10:
    print("✓ PASS - GPU constraint satisfied exactly")
else:
    print("✗ FAIL - GPU constraint not satisfied")