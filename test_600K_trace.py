#!/usr/bin/env python
"""Test only the 600K case and capture detailed trace."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Load database
dbf = Database('NbTi.tdb')

# Test parameters for 600K case only
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 600, v.P: 101325, v.X('TI'): 0.1, v.N: 1}

print("Testing 600K, X(TI)=0.1 case...")

# CPU calculation
print("\n=== CPU CALCULATION ===")
cpu_result = equilibrium(dbf, comps, phases, conds, 
                        calc_opts={'pdens': 50})

# GPU calculation  
print("\n=== GPU CALCULATION ===")
gpu_result = equilibrium(dbf, comps, phases, conds, 
                        calc_opts={'pdens': 50}, gpu=True)

# Extract results
cpu_gm = float(cpu_result.GM.values)
gpu_gm = float(gpu_result.GM.values)

cpu_mu = {
    'NB': float(cpu_result.MU.sel(component='NB').values),
    'TI': float(cpu_result.MU.sel(component='TI').values)
}
gpu_mu = {
    'NB': float(gpu_result.MU.sel(component='NB').values),
    'TI': float(gpu_result.MU.sel(component='TI').values)
}

# Calculate differences
gm_diff = abs(cpu_gm - gpu_gm)
mu_nb_diff = abs(cpu_mu['NB'] - gpu_mu['NB'])
mu_ti_diff = abs(cpu_mu['TI'] - gpu_mu['TI'])

print(f"\n=== RESULTS ===")
print(f"CPU GM: {cpu_gm:.9f}")
print(f"GPU GM: {gpu_gm:.9f}")
print(f"GM difference: {gm_diff:.9f}")
print(f"\nCPU MU(NB): {cpu_mu['NB']:.9f}")
print(f"GPU MU(NB): {gpu_mu['NB']:.9f}")
print(f"MU(NB) difference: {mu_nb_diff:.9f}")
print(f"\nCPU MU(TI): {cpu_mu['TI']:.9f}")
print(f"GPU MU(TI): {gpu_mu['TI']:.9f}")
print(f"MU(TI) difference: {mu_ti_diff:.9f}")

if gm_diff < 1e-6:
    print("\n✓ PASS: Differences are within tolerance")
else:
    print("\n✗ FAIL: Differences exceed tolerance")