#!/usr/bin/env python
"""Test to see what constraint values the GPU is using."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Load database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'FCC_A1']

# Single ternary condition
conditions = {
    v.T: 1200,
    v.P: 101325,
    v.X('CU'): 0.2,  
    v.X('FE'): 0.3   
}

print("Testing constraint calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False, calc_opts={'pdens': 50})