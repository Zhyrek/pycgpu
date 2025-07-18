#!/usr/bin/env python
"""Trace CPU RHS calculation in detail."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Let's run CPU with verbose output to see the consolidation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("CPU EQUILIBRIUM WITH VERBOSE OUTPUT")
print("=" * 60)

# Run with verbose to see the consolidation and RHS values
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)

cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")
print(f"Number of phases: {sum(1 for np in cpu_np if np > 1e-12)}")

# Look at the single phase details
for i, (np_val, x_ti) in enumerate(zip(cpu_np, cpu_x_ti)):
    if np_val > 1e-12:
        print(f"Phase {i}: NP={np_val:.6f}, X(TI)={x_ti:.10f}")