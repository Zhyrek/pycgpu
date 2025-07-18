#!/usr/bin/env python
"""Force regeneration of GPU kernel with updated solve_state naming."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Delete cached kernel to force regeneration
import os
kernel_path = '/mnt/c/users/scott/Documents/pycalphad/generated_equilibrium_kernel.cu'
if os.path.exists(kernel_path):
    print(f"Removing existing kernel: {kernel_path}")
    os.remove(kernel_path)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Running GPU equilibrium to regenerate kernel...")
try:
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()[0]
    print(f"GPU X(TI): {gpu_x_ti:.10f}")
except Exception as e:
    print(f"Error: {e}")
    print("This may be expected if kernel generation failed.")

print("\nCheck generated_equilibrium_kernel.cu for 'solve_state_global_mem' references.")
print("There should be NONE - all should be renamed to 'solve_state'.")