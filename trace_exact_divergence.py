#!/usr/bin/env python3
"""Trace exact point where CPU and GPU diverge"""
import os
os.environ['PYCALPHAD_DEBUG'] = '1'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear cached GPU modules
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Tracing CPU vs GPU Divergence ===")
print("\nKey values to compare:")
print("1. Initial phase energies")
print("2. Initial Hessian values")
print("3. Initial gradient values")
print("4. c_G values (phase matrix)")
print("5. Matrix coefficients in equilibrium system")
print("6. Solution vector from linear solver")

# Run calculations to capture debug output
print("\n" + "="*60)
print("CPU CALCULATION")
print("="*60)
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})

print("\n" + "="*60)
print("GPU CALCULATION")
print("="*60)
gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})

cpu_gm = float(cpu_result.GM.values.flatten()[0])
gpu_gm = float(gpu_result.GM.values.flatten()[0])

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"CPU GM: {cpu_gm:.1f} J/mol")
print(f"GPU GM: {gpu_gm:.1f} J/mol")
print(f"Difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")