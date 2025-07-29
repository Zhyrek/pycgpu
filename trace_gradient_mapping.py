#!/usr/bin/env python
"""Trace gradient mapping issue in detail."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Single test condition
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Tracing gradient mapping issue...")
print("="*80)

# Run CPU calculation
cpu_result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=False, gpu=False)

cpu_gm = float(cpu_result.GM.values)
cpu_mu_nb = float(cpu_result.MU.sel(component='NB').values)
cpu_mu_ti = float(cpu_result.MU.sel(component='TI').values)

print(f"CPU Results:")
print(f"  GM: {cpu_gm}")
print(f"  MU(NB): {cpu_mu_nb}")
print(f"  MU(TI): {cpu_mu_ti}")
print()

# Run GPU calculation with more debug output
import os
os.environ['PYCALPHAD_GPU_DEBUG'] = '1'

gpu_result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=False, gpu=True)

gpu_gm = float(gpu_result.GM.values)
gpu_mu_nb = float(gpu_result.MU.sel(component='NB').values)
gpu_mu_ti = float(gpu_result.MU.sel(component='TI').values)

print(f"\nGPU Results:")
print(f"  GM: {gpu_gm}")
print(f"  MU(NB): {gpu_mu_nb}")
print(f"  MU(TI): {gpu_mu_ti}")
print()

print(f"Differences:")
print(f"  GM diff: {abs(cpu_gm - gpu_gm)}")
print(f"  MU(NB) diff: {abs(cpu_mu_nb - gpu_mu_nb)}")
print(f"  MU(TI) diff: {abs(cpu_mu_ti - gpu_mu_ti)}")

print("="*80)