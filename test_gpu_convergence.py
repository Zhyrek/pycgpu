#!/usr/bin/env python3
"""Test if GPU converges with mass jacobian fix"""
import os
os.environ['PYCALPHAD_DEBUG'] = '0'  # Turn off debug for cleaner output

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

print("=== Testing GPU Convergence ===")

# Run CPU calculation
print("\nCPU calculation:")
cpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=False, calc_opts={'pdens': 50})
cpu_gm = float(cpu_result.GM.values.flatten()[0])
cpu_phases = [p for p in cpu_result.Phase.values.flatten() if p != '']
cpu_np = cpu_result.NP.values.flatten()
cpu_active = cpu_np[cpu_np > 1e-10]
print(f"GM: {cpu_gm:.1f} J/mol")
print(f"Phases: {cpu_phases}")
print(f"Active phase amounts: {cpu_active}")

# Run GPU calculation
print("\nGPU calculation:")
gpu_result = equilibrium(db, comps, phases, conditions, verbose=False, gpu=True, calc_opts={'pdens': 50})
gpu_gm = float(gpu_result.GM.values.flatten()[0])
gpu_phases = [p for p in gpu_result.Phase.values.flatten() if p != '']
gpu_np = gpu_result.NP.values.flatten()
gpu_active = gpu_np[gpu_np > 1e-10]
print(f"GM: {gpu_gm:.1f} J/mol")
print(f"Phases: {gpu_phases}")
print(f"Active phase amounts: {gpu_active}")

# Compare
print(f"\n=== Comparison ===")
print(f"GM difference: {abs(gpu_gm - cpu_gm):.1f} J/mol")
print(f"Number of active phases - CPU: {len(cpu_active)}, GPU: {len(gpu_active)}")

if len(gpu_active) < len(cpu_active):
    print("\n⚠️  GPU removed phases that CPU kept!")
    print("This explains the energy difference")
else:
    print("\n✓ GPU kept same number of phases as CPU")