#!/usr/bin/env python3
"""Debug why CPU gets 2 phases initially"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== CPU 2-Phase Debug ===\n")

# Run CPU equilibrium and capture debug output
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

# Look for key information
print("Looking for phase information in CPU output:")
for line in output.split('\n'):
    if 'Phase counts' in line or 'num_phases' in line or 'CONSOLIDAT' in line:
        print(f"  {line}")

print(f"\nCPU Final Result:")
print(f"  GM: {eq_cpu.GM.values[0]:.1f} J/mol")
print(f"  Active phases: {[p for p in eq_cpu.Phase.values.flat if p != '']}")
print(f"  Phase amounts: {eq_cpu.NP.values[eq_cpu.NP.values > 1e-6]}")