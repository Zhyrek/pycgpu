#!/usr/bin/env python
"""Simple GPU test"""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Simple binary system for testing
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3', 'LIQUID']

# Single condition for easy comparison
conds = {v.T: 1000, v.P: 101325, v.X('TI'): 0.4}

print("=== SIMPLE GPU TEST ===")

# Run GPU calculation
try:
    gpu_result = equilibrium(dbf, comps, phases, conds, verbose=False, gpu=True)
    print(f"GPU GM: {gpu_result.GM.values.flatten()[0]:.6f} J/mol")
except Exception as e:
    print(f"GPU failed: {e}")
    
# Run CPU calculation  
cpu_result = equilibrium(dbf, comps, phases, conds, verbose=False, gpu=False)
print(f"CPU GM: {cpu_result.GM.values.flatten()[0]:.6f} J/mol")