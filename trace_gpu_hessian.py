#!/usr/bin/env python3
"""Trace GPU Hessian values"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== GPU Hessian Values ===\n")

# Run GPU
eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                gpu=True, verbose=True)