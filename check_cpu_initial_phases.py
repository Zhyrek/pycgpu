#!/usr/bin/env python
"""Check what initial phases CPU has."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("CHECKING CPU INITIAL PHASES")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nRunning CPU equilibrium...")

try:
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
    print(f"CPU final result: X(TI) = {cpu_x_ti:.10f}")
except Exception as e:
    print(f"CPU failed: {e}")

print("\nThe key question:")
print("Do CPU and GPU start with the same initial phase compositions?")
print("If not, that could explain why they consolidate differently.")