#!/usr/bin/env python3
"""Simple test to see what starting_point returns"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Starting Point Test ===\n")

# First, let's run CPU equilibrium and capture its debug output
print("1. Running CPU equilibrium to see what it does:")

import sys
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

# Run equilibrium
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    verbose=True)

# Get output
output = mystdout.getvalue()
sys.stdout = old_stdout

# Look for key information about starting point
for line in output.split('\n'):
    if 'Starting point' in line or 'Result GM' in line or 'Active phase' in line or 'Lower convex' in line:
        print(f"   {line.strip()}")

print(f"\n2. CPU Final Result:")
print(f"   GM: {eq_cpu.GM.values[0]:.1f} J/mol")

# Count phases at different points
phase_count = 0
for p in eq_cpu.Phase.values.flat:
    if p != '' and p != '_FAKE_':
        phase_count += 1
print(f"   Total phase slots with values: {phase_count}")
print(f"   Phase array: {eq_cpu.Phase.values}")
print(f"   NP array: {eq_cpu.NP.values}")