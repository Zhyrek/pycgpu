#!/usr/bin/env python3
"""Compare c_G values between CPU and GPU at iteration 0"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Comparing c_G values at iteration 0 ===\n")

# Run CPU and capture c_G from the mole fraction RHS debug
print("Looking for c_G values in mole fraction constraint calculation...")
print("\nThese are the c_G values being used in the RHS calculation:")
print("(From 'c_G values:' lines in CPU MOLE FRAC RHS DEBUG output)\n")

eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                verbose=True)