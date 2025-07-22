#!/usr/bin/env python
"""Detailed test of single condition X(TI)=0.1, T=600K with verbose output."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import cupy as cp
import os
import shutil

# Clear CuPy kernel cache to ensure code changes take effect
cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared CuPy kernel cache: {cache_dir}")

# Clear CuPy memory cache
cp.cuda.runtime.memGetInfo()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing X(TI)=0.1, T=600K with verbose output")
print("="*60)

# CPU result
print("\n=== CPU CALCULATION ===")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
gm_cpu = float(result_cpu.GM.values)
print(f"\nFinal CPU GM: {gm_cpu:.9f}")

# Analyze CPU phases
cpu_phases = result_cpu.Phase.values.flatten()
cpu_np = result_cpu.NP.values.flatten()
print(f"CPU phases present: {[p for p, n in zip(cpu_phases, cpu_np) if n > 1e-10]}")
print(f"CPU phase amounts: {[n for n in cpu_np if n > 1e-10]}")

# GPU result
print("\n\n=== GPU CALCULATION ===")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
gm_gpu = float(result_gpu.GM.values)
print(f"\nFinal GPU GM: {gm_gpu:.9f}")

# Analyze GPU phases
gpu_phases = result_gpu.Phase.values.flatten()
gpu_np = result_gpu.NP.values.flatten()
print(f"GPU phases present: {[p for p, n in zip(gpu_phases, gpu_np) if n > 1e-10]}")
print(f"GPU phase amounts: {[n for n in gpu_np if n > 1e-10]}")

# Compare
print(f"\n=== COMPARISON ===")
print(f"GM difference: {abs(gm_gpu - gm_cpu):.9f}")
print(f"PASS" if abs(gm_gpu - gm_cpu) < 0.001 else f"FAIL")

# Check chemical potentials
mu_cpu = result_cpu.MU.values.flatten()[:2]
mu_gpu = result_gpu.MU.values.flatten()[:2]
print(f"\nChemical potentials:")
print(f"  CPU: μ(NB)={mu_cpu[0]:.9f}, μ(TI)={mu_cpu[1]:.9f}")
print(f"  GPU: μ(NB)={mu_gpu[0]:.9f}, μ(TI)={mu_gpu[1]:.9f}")
print(f"  Differences: μ(NB)={abs(mu_gpu[0]-mu_cpu[0]):.9f}, μ(TI)={abs(mu_gpu[1]-mu_cpu[1]):.9f}")

# Check site fractions
print(f"\nSite fractions:")
for phase, np_cpu, np_gpu in zip(['BCC_A2#1', 'BCC_A2#2'], cpu_np, gpu_np):
    if np_cpu > 1e-10 or np_gpu > 1e-10:
        print(f"  {phase}:")
        print(f"    CPU amount: {np_cpu:.9f}")
        print(f"    GPU amount: {np_gpu:.9f}")