#!/usr/bin/env python
"""Detailed trace of iterations for debugging CPU/GPU differences."""

import numpy as np
from pycalphad import Database, equilibrium
import pycalphad.variables as v

# Load database
dbf = Database('NbTi.tdb')

# Test parameters for 600K case only
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']
conds = {v.T: 600, v.P: 101325, v.X('TI'): 0.1, v.N: 1}

print("Testing 600K, X(TI)=0.1 case with detailed iteration trace...")

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

# Extract phase information
cpu_phase_amt = cpu_result.NP.values
gpu_phase_amt = gpu_result.NP.values

cpu_y_nb = cpu_result.Y.sel(vertex=0, component='NB').values
cpu_y_ti = cpu_result.Y.sel(vertex=0, component='TI').values
gpu_y_nb = gpu_result.Y.sel(vertex=0, component='NB').values  
gpu_y_ti = gpu_result.Y.sel(vertex=0, component='TI').values

# Calculate differences
gm_diff = abs(cpu_gm - gpu_gm)
mu_nb_diff = abs(cpu_mu['NB'] - gpu_mu['NB'])
mu_ti_diff = abs(cpu_mu['TI'] - gpu_mu['TI'])

print(f"\n=== FINAL RESULTS ===")
print(f"CPU: GM={cpu_gm:.9f}, MU(NB)={cpu_mu['NB']:.9f}, MU(TI)={cpu_mu['TI']:.9f}")
print(f"GPU: GM={gpu_gm:.9f}, MU(NB)={gpu_mu['NB']:.9f}, MU(TI)={gpu_mu['TI']:.9f}")
print(f"Differences: GM={gm_diff:.9f}, MU(NB)={mu_nb_diff:.9f}, MU(TI)={mu_ti_diff:.9f}")

print(f"\nPhase information:")
print(f"CPU: NP={cpu_phase_amt}, Y(NB)={cpu_y_nb}, Y(TI)={cpu_y_ti}")
print(f"GPU: NP={gpu_phase_amt}, Y(NB)={gpu_y_nb}, Y(TI)={gpu_y_ti}")

if gm_diff < 1e-6:
    print("\n✓ PASS: Differences are within tolerance")
else:
    print("\n✗ FAIL: Differences exceed tolerance")