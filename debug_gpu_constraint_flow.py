#!/usr/bin/env python
"""Debug GPU constraint flow to see where the constraint gets lost."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing X(TI) = 0.005 to trace constraint flow...")

conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

# Run GPU with focused debug output
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)