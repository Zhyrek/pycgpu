#!/usr/bin/env python3
"""Debug phase amount updates in GPU solver"""

import sys
sys.path.insert(0, '.')

from pycalphad import Database, equilibrium, variables as v
from pycalphad.gpu.gpu_equilibrium import clear_gpu_cache

# Clear GPU cache to ensure fresh functions
clear_gpu_cache()

# Load database
db = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'LIQUID']
eq_conditions = {v.N: 1, v.P: 101325, v.T: 1000, v.X('TI'): 0.4}

print("=== Testing GPU Phase Updates ===")

# Run GPU calculation with maximum verbosity
print("\n=== GPU Calculation ===")
gpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=True, gpu=True, to='GM', calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values)
gpu_np = gpu_result.NP.values
gpu_phase = gpu_result.Phase.values

print(f"\nGPU Final State:")
print(f"  GM: {gpu_gm:.6f} J/mol")
print(f"  Phase amounts: {gpu_np}")
print(f"  Phases: {gpu_phase}")

# Run CPU for comparison
print("\n=== CPU Calculation (for reference) ===")
cpu_result = equilibrium(db, comps, phases, eq_conditions, verbose=False, gpu=False, to='GM', calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values)
cpu_np = cpu_result.NP.values
cpu_phase = cpu_result.Phase.values

print(f"\nCPU Final State:")
print(f"  GM: {cpu_gm:.6f} J/mol")
print(f"  Phase amounts: {cpu_np}")
print(f"  Phases: {cpu_phase}")

print(f"\n=== Comparison ===")
print(f"GM difference: {abs(gpu_gm - cpu_gm):.6f} J/mol")
print(f"GPU/CPU ratio: {gpu_gm / cpu_gm:.6f}")