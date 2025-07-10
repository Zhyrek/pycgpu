#!/usr/bin/env python3
"""Check if GPU Hessian fix is being applied correctly"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Checking GPU Hessian Fix ===\n")
print("The fix_hessian_spurious_terms function should remove spurious entropy cross-terms")
print("from diagonal Hessian elements, which were causing GPU values to be ~2.5x larger.\n")

# Run GPU with verbose to see if fix is being applied
eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                gpu=True, verbose=True)

print("\nIf the fix is working, we should see:")
print("1. '[GPU HESSIAN FIX] Processing diagonal element' messages")
print("2. Messages about removing spurious terms")
print("3. GPU Hessian values matching CPU values")