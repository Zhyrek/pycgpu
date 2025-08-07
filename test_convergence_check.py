#!/usr/bin/env python
"""Check which threads are not converging."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

dbf = Database('AuBi-07Wan.tdb')
comps = ['AU', 'BI', 'VA']
phases = ['LIQUID', 'FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7', 'BCC_A2']

x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
temp_values = [400, 500, 600, 700]

print("Running GPU calculation with verbose output...")
result_gpu = equilibrium(dbf, comps, phases, 
                        {v.X('BI'): x_bi_values, v.T: temp_values, v.P: 101325}, 
                        gpu=True, verbose=True)

# The verbose output should show which threads converged