#!/usr/bin/env python3
"""Trace the source of the 2.5x Hessian factor"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Tracing the 2.5x Hessian Factor ===\n")

# At the starting point:
# Phase 0: Y_NB = 0.6, Y_TI = 0.4
# Phase 1: Y_NB = 0.5, Y_TI = 0.5

print("Starting compositions:")
print("Phase 0: Y_NB = 0.6, Y_TI = 0.4, sum = 1.0")
print("Phase 1: Y_NB = 0.5, Y_TI = 0.5, sum = 1.0")
print()

# The GPU Hessian is 2.5x larger
# Let's check: 1.0 / 0.4 = 2.5
print("Interesting observation: 1.0 / Y_TI = 1.0 / 0.4 = 2.5")
print("This matches the factor we're seeing!")
print()

# This suggests the GPU might be calculating:
# d²G/dY_NB² + extra_term_with_1/Y_TI

print("Hypothesis: The GPU Hessian includes an extra term with 1/Y_TI")
print("that wasn't fully removed by fix_hessian_spurious_terms.")
print()

print("For Phase 1 with Y_TI = 0.5:")
print("Expected factor: 1.0 / 0.5 = 2.0")
print("Let's check the actual GPU values...")

# Run equilibrium to see the values
eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                gpu=True, verbose=True)