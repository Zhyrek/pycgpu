#!/usr/bin/env python
"""Debug CPU site fraction updates at iteration 1."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("DEBUGGING CPU SITE FRACTION UPDATES")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning CPU calculation with verbose output...")
print("\nLooking for:")
print("1. Site fraction values at iteration 1 (after consolidation)")
print("2. delta_y calculations at iteration 1")
print("3. Site fraction updates applied at iteration 1")
print("4. Site fraction values at iteration 2")

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    print("\nCPU completed")
except Exception as e:
    print(f"CPU error: {e}")