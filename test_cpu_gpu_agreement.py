#!/usr/bin/env python
"""Test CPU vs GPU agreement after fixes."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

# Clear cache to force recompilation
import subprocess
subprocess.run(['rm', '-f', '/home/user/.cache/pycalphad/*.cu'], capture_output=True)

# Run calculations
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

# Extract results
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]

print(f"CPU X(TI): {cpu_x_ti:.10f}")
print(f"GPU X(TI): {gpu_x_ti:.10f}")
print(f"Absolute difference: {abs(cpu_x_ti - gpu_x_ti):.10f}")
print(f"Relative error: {abs(cpu_x_ti - gpu_x_ti)/cpu_x_ti * 100:.6f}%")