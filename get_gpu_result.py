#!/usr/bin/env python
"""Get just the final GPU result."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic case
conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
x_ti_final = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"Final GPU X(TI) = {x_ti_final:.12f}")
print(f"Expected X(TI) = 0.005000000000")
print(f"Error = {x_ti_final - 0.005:.2e}")

# Check if this exact value matches our problematic value
if abs(x_ti_final - 0.01028221) < 1e-8:
    print("✓ This matches the problematic value 0.01028221")
else:
    print(f"✗ This does not match 0.01028221 (diff = {abs(x_ti_final - 0.01028221):.2e})")