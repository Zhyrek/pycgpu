#!/usr/bin/env python
"""Test the fix for consolidated phase amounts."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K (single-phase region)
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing fix for X(TI)=0.1, T=600K (single-phase region)")
print("="*80)

# Run GPU calculation
print("\nGPU Calculation:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"GPU GM: {gpu_gm:.6f} J/mol")
print(f"GPU Phases: {gpu_phases}")
print(f"GPU NP: {gpu_np}")

# Count stable phases
gpu_stable_count = 0
for phase, amt in zip(gpu_phases, gpu_np):
    if phase and amt > 1e-6:
        gpu_stable_count += 1
print(f"GPU stable phases: {gpu_stable_count}")

# Run CPU calculation
print("\nCPU Calculation:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print(f"CPU GM: {cpu_gm:.6f} J/mol")
print(f"CPU Phases: {cpu_phases}")
print(f"CPU NP: {cpu_np}")

# Count stable phases
cpu_stable_count = 0
for phase, amt in zip(cpu_phases, cpu_np):
    if phase and amt > 1e-6:
        cpu_stable_count += 1
print(f"CPU stable phases: {cpu_stable_count}")

# Compare results
print("\nComparison:")
print("-"*40)
print(f"GM error: {gpu_gm - cpu_gm:.6f} J/mol")
print(f"Expected error: ~0 J/mol (< 0.001 J/mol)")

if abs(gpu_gm - cpu_gm) < 0.001:
    print("\n✓ SUCCESS! GPU and CPU results match within tolerance.")
else:
    print(f"\n✗ FAIL! Error of {abs(gpu_gm - cpu_gm):.6f} J/mol exceeds tolerance.")

# Also test a two-phase region to make sure we didn't break it
print("\n\nTesting two-phase region (X(TI)=0.5, T=600K):")
print("="*80)

conditions_2phase = {v.X('TI'): 0.5, v.T: 600, v.P: 101325}

result_gpu_2phase = equilibrium(dbf, comps, phases, conditions_2phase, gpu=True, verbose=False)
result_cpu_2phase = equilibrium(dbf, comps, phases, conditions_2phase, gpu=False, verbose=False)

gpu_gm_2phase = float(result_gpu_2phase.GM.values)
cpu_gm_2phase = float(result_cpu_2phase.GM.values)

print(f"GPU GM: {gpu_gm_2phase:.6f} J/mol")
print(f"CPU GM: {cpu_gm_2phase:.6f} J/mol")
print(f"Error: {gpu_gm_2phase - cpu_gm_2phase:.6f} J/mol")

if abs(gpu_gm_2phase - cpu_gm_2phase) < 0.001:
    print("✓ Two-phase region still works correctly.")
else:
    print(f"✗ Two-phase region broken! Error: {abs(gpu_gm_2phase - cpu_gm_2phase):.6f} J/mol")