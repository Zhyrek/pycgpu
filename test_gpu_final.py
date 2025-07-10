#!/usr/bin/env python3
"""Final comparison test without debug output"""
import numpy as np
from pycalphad import Database, equilibrium, variables as v
import sys
import os

# Disable debug output
os.environ['PYCALPHAD_DEBUG'] = '0'

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']

# Set conditions
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("Testing GPU vs CPU equilibrium calculation...")
print("Conditions: T=1000K, X(TI)=0.4")
print()

# Run CPU calculation
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
print(f"CPU GM: {cpu_gm:.6f} J/mol")

# Run GPU calculation
gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])
print(f"GPU GM: {gpu_gm:.6f} J/mol")

# Compare results
print(f"\nDifference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
print(f"Match within 0.001 J tolerance: {abs(gpu_gm - cpu_gm) < 0.001}")

# Show phase information
print("\nPhase information:")
print("CPU phases:", [p for p in cpu_result.Phase.values.flatten() if p != ''])
print("GPU phases:", [p for p in gpu_result.Phase.values.flatten() if p != ''])