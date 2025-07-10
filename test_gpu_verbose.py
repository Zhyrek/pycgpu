#!/usr/bin/env python3
"""Test GPU with verbose output"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions from original test
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== GPU Verbose Test ===\n")

# Run GPU equilibrium with verbose=True
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

print(f"\nGPU Result: GM = {eq_gpu.GM.values[0]:.1f} J/mol")