#!/usr/bin/env python
"""Test phase amount sum behavior for consolidation."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

print("Testing phase amount sum behavior")
print("="*80)

# Test case that requires consolidation
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

# Run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
cpu_active_amounts = [a for a in cpu_np if a > 1e-6]
cpu_phase_sum = sum(cpu_active_amounts)

print("CPU Result:")
print(f"  GM = {cpu_gm:.15f} J/mol")
print(f"  Active phase amounts: {cpu_active_amounts}")
print(f"  Sum of phase amounts: {cpu_phase_sum:.15f}")
print(f"  Is sum exactly 1.0? {cpu_phase_sum == 1.0}")
print()

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()
gpu_active_amounts = [a for a in gpu_np if a > 1e-6]
gpu_phase_sum = sum(gpu_active_amounts)

print("GPU Result:")
print(f"  GM = {gpu_gm:.15f} J/mol")
print(f"  Active phase amounts: {gpu_active_amounts}")
print(f"  Sum of phase amounts: {gpu_phase_sum:.15f}")
print(f"  Is sum exactly 1.0? {gpu_phase_sum == 1.0}")
print()

# Analysis
print("Analysis:")
print(f"  CPU phase sum deviation from 1.0: {abs(cpu_phase_sum - 1.0):.15e}")
print(f"  GPU phase sum deviation from 1.0: {abs(gpu_phase_sum - 1.0):.15e}")
print(f"  GM error: {gpu_gm - cpu_gm:.15e} J/mol")

# Calculate what GM would be if GPU used CPU's phase amount
if len(gpu_active_amounts) == 1 and len(cpu_active_amounts) == 1:
    # Assuming single phase with same energy
    gpu_energy_per_mole = gpu_gm / gpu_phase_sum  # Energy per formula unit
    cpu_like_gm = gpu_energy_per_mole * cpu_phase_sum
    print(f"\nIf GPU used CPU's phase amount:")
    print(f"  Estimated GM = {cpu_like_gm:.15f} J/mol")
    print(f"  This would give error: {cpu_like_gm - cpu_gm:.15e} J/mol")