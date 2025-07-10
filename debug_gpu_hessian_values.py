#!/usr/bin/env python3
"""Debug GPU Hessian values to understand 2.5x factor"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Debugging GPU Hessian 2.5x Issue ===\n")
print("The fix_hessian_spurious_terms IS being applied correctly.")
print("But GPU Hessian values are still 2.5x larger than CPU.\n")

# Let's check specific values
print("Expected behavior:")
print("- CPU Phase 0: hess[3,3] = 4.157250e+03")
print("- GPU Phase 0: hess[3,3] = 1.039313e+04 (2.5x larger)")
print("\nLet's trace where this factor comes from...\n")

# Run GPU to see actual Hessian values
eq = equilibrium(db, comps, phases, conditions,
                calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                gpu=True, verbose=True)

print("\nPossible sources of 2.5x factor:")
print("1. Different normalization of site fractions")
print("2. Different handling of (Y1 + Y2) denominator")
print("3. Different entropy contribution calculation")
print("4. Model vs Workspace DOF format conversion issue")