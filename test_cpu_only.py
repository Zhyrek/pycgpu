#!/usr/bin/env python3
"""Test CPU equilibrium calculation only"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing CPU Equilibrium ===")

# Run CPU calculation
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, to='GM', calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values)

print(f"\nCPU GM: {cpu_gm:.6f} J/mol")