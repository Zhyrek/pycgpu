#!/usr/bin/env python
"""Trace where specific GPU values come from."""

import os
import sys
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Let's test the X(TI) = 0.005 case that gives us 0.01028221
print("Tracing GPU execution for X(TI) = 0.005 → 0.01028221...")

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

conditions = {v.X('TI'): 0.005, v.T: 1000, v.P: 101325}

# Run GPU and capture output to look for where 0.01028221 appears
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)