#!/usr/bin/env python
"""Test GPU vs CPU after SVD fix."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Clear cache to force recompilation
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("Testing X(TI)=0.9, T=600K")
print("=" * 60)

# Run CPU
print("\nCPU calculation:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_x_ti = result_cpu.X.sel(component='TI').values.flatten()
cpu_np = result_cpu.NP.values.flatten()
overall_x_ti_cpu = sum(np_val * x_ti for np_val, x_ti in zip(cpu_np, cpu_x_ti) if np_val > 1e-12)
print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU X(TI): {overall_x_ti_cpu:.10f}")

# Run GPU
print("\nGPU calculation:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()
overall_x_ti_gpu = sum(np_val * x_ti for np_val, x_ti in zip(gpu_np, gpu_x_ti) if np_val > 1e-12)
print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"GPU X(TI): {overall_x_ti_gpu:.10f}")

# Compare
print(f"\nError: {abs(gpu_gm - cpu_gm):.6f} J/mol")
print(f"X(TI) error: {abs(overall_x_ti_gpu - 0.9):.10f}")