#!/usr/bin/env python3
"""Trace where GPU loses the second phase"""

from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load database  
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2']

# Set conditions
conditions = {v.N: 1, v.P: 101325, v.T: 300, v.X('TI'): 0.4}

print("=== Tracing GPU Phase Loss ===\n")

# First, let's see what CPU equilibrium does
print("1. CPU Equilibrium (for reference):")
eq_cpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}})

cpu_phases = [p for p in eq_cpu.Phase.values.flat if p != '']
cpu_np = eq_cpu.NP.values[eq_cpu.NP.values > 1e-6]
print(f"   CPU phases: {cpu_phases}")  
print(f"   CPU phase amounts: {cpu_np}")
print(f"   CPU GM: {eq_cpu.GM.values[0]:.1f} J/mol")

# Now trace GPU with verbose
print("\n2. GPU Equilibrium (verbose):")
eq_gpu = equilibrium(db, comps, phases, conditions,
                    calc_opts={'points': {'BCC_A2': [[0.6, 0.4], [0.5, 0.5]]}},
                    gpu=True, verbose=True)

print(f"\n3. GPU Result:")
gpu_phases = [p for p in eq_gpu.Phase.values.flat if p != '']
gpu_np = eq_gpu.NP.values[eq_gpu.NP.values > 1e-6]
print(f"   GPU phases: {gpu_phases}")
print(f"   GPU phase amounts: {gpu_np}")
print(f"   GPU GM: {eq_gpu.GM.values[0]:.1f} J/mol")