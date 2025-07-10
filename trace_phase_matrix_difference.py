#!/usr/bin/env python3
"""Trace phase matrix construction difference between CPU and GPU"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Tracing Phase Matrix Construction ===\n")

# The phase matrix for a binary substitutional solution should be:
# [H11  H12  1]
# [H21  H22  1]  
# [1    1    0]
#
# Where H is the Hessian of the energy w.r.t. site fractions
# and the constraint is Y(NB) + Y(TI) = 1

print("Expected phase matrix structure for binary substitutional solution:")
print("[H(Y1,Y1)  H(Y1,Y2)  1]")
print("[H(Y2,Y1)  H(Y2,Y2)  1]")  
print("[1         1         0]")
print("\nLet's see what CPU and GPU actually construct...\n")

# Run CPU
eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                verbose=True)