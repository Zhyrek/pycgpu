#!/usr/bin/env python
"""Detailed test of one AlCu condition with dgelsd to trace divergence."""

import os
os.environ['PYCALPHAD_DEBUG_EQSOLVER'] = '1'  # Enable debug output

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load AlCuFe database
dbf = Database('Al-Cu-Fe.tdb')
comps = ['AL', 'CU', 'FE', 'VA']
phases = ['LIQUID', 'ALCU_ZETA']

print("Detailed test: Med T, balanced condition")
print("X(AL)=0.6, X(CU)=0.3, X(FE)=0.1, T=900K")
print("="*70)

conditions = {
    v.T: 900, 
    v.P: 101325, 
    v.N: 1, 
    v.X('AL'): 0.6,
    v.X('CU'): 0.3
}

# Run CPU calculation
print("\n" + "="*70)
print("CPU CALCULATION")
print("="*70)
cpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False)
cpu_gm = float(cpu_result.GM.values.item())
print(f"\nCPU Final GM: {cpu_gm:.6f} J/mol")

# Clear output
print("\n" + "="*70)
print("GPU CALCULATION")  
print("="*70)

# Run GPU calculation
gpu_result = equilibrium(dbf, comps, phases, conditions, 
                       calc_opts={'pdens': 100}, verbose=False, gpu=True)
gpu_gm = float(gpu_result.GM.values.item())
print(f"\nGPU Final GM: {gpu_gm:.6f} J/mol")

# Compare
diff = abs(cpu_gm - gpu_gm)
print(f"\n" + "="*70)
print(f"DIFFERENCE: {diff:.2e} J/mol")

# Check if this is better than before
print(f"\nPrevious error for this condition: ~6.94 J/mol")
print(f"Current error with dgelsd: {diff:.2e} J/mol")
if diff < 6.94:
    print("✓ IMPROVED!")
else:
    print("✗ No improvement or worse")
    print("\nThe dgelsd implementation alone is not sufficient.")
    print("Need to investigate other sources of divergence.")