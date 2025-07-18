#!/usr/bin/env python
"""Test GPU c_G values after consolidation."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("GPU EQUILIBRIUM - FOCUS ON c_G AFTER CONSOLIDATION")
print("=" * 60)

# Run GPU equilibrium
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)

print(f"\nFINAL: X(TI) = {overall_x_ti:.10f}")
print(f"Number of phases: {sum(1 for np in gpu_np if np > 1e-12)}")