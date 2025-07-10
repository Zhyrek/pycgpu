#!/usr/bin/env python3
"""Test GPU phase compositions"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions from original test
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== GPU Phase Compositions Test ===\n")

# Run GPU equilibrium only
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True)

print(f"\nGPU Result: GM = {eq_gpu.GM.values[0]:.1f} J/mol")