#!/usr/bin/env python
"""Test gradient ordering output."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Single test condition
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Testing gradient ordering...")
print("="*80)

# Run GPU calculation to trigger gradient generation
gpu_result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=False, gpu=True)

print("\nDone.")
print("="*80)