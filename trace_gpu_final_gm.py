#!/usr/bin/env python
"""Trace GPU final GM calculation in detail."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import cupy as cp

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing GPU final GM calculation for X(TI)=0.1, T=600K")
print("="*80)

# First clear CuPy cache to ensure fresh kernel compilation
print("\nClearing CuPy cache...")
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
# Clear kernel cache
import os
import shutil
cache_dir = os.path.expanduser('~/.cupy/kernel_cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared CuPy kernel cache at {cache_dir}")

# Run GPU calculation
print("\nRunning GPU calculation...")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)

# Extract results
gpu_gm = float(result_gpu.GM.values)
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()

print(f"\nGPU Results:")
print(f"GM: {gpu_gm:.6f} J/mol")
print(f"Phases: {gpu_phases}")
print(f"NP: {gpu_np}")

# Count stable phases
stable_phases = []
for i, (phase, amt) in enumerate(zip(gpu_phases, gpu_np)):
    if phase and amt > 1e-6:
        stable_phases.append((i, phase, amt))

print(f"\nStable phases: {len(stable_phases)}")
for i, phase, amt in stable_phases:
    print(f"  Phase {i}: {phase}, amount={amt:.6f}")

# Run CPU for comparison
print("\n\nCPU Results for comparison:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = float(result_cpu.GM.values)
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()

print(f"GM: {cpu_gm:.6f} J/mol")
print(f"Phases: {cpu_phases}")  
print(f"NP: {cpu_np}")

print(f"\nError: {gpu_gm - cpu_gm:.6f} J/mol")

# Analysis
print("\n\nAnalysis:")
print("-"*80)
print("The GPU is reporting 2 BCC_A2 phases with amounts [0.5, 0.5]")
print("This means the GPU final GM calculation is:")
print(f"  GM = 0.5 * energy_phase0 + 0.5 * energy_phase1")
print("\nBut after consolidation, it should be:")
print(f"  GM = 1.0 * energy_single_phase")
print("\nThe issue is that the GPU is using the wrong phase amounts in the final GM calculation.")