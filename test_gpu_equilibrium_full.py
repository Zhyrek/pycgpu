#!/usr/bin/env python
"""Test full GPU equilibrium to see how kernel is generated."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database and set up system
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = list(db.phases.keys())

print("Phases:", phases)

# Run equilibrium with GPU - this should generate the kernel
T = 1300.0
conditions = {v.T: T, v.P: 101325, v.X('TI'): 0.5}

print("\nRunning equilibrium with GPU=True...")
eq_result = equilibrium(db, comps, phases, conditions, 
                       output=['GM'], gpu=True, 
                       calc_opts={'N': 1},
                       verbose=True)

print("\nEquilibrium completed")