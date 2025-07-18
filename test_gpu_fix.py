#!/usr/bin/env python
"""Test GPU with fix to match CPU by treating all site fractions as independent."""

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

print("TESTING GPU FIX - ALL SITE FRACTIONS AS INDEPENDENT")
print("=" * 60)

dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the problematic condition
conditions = {v.X('TI'): 0.9, v.T: 600, v.P: 101325}

print("\nTest condition: X(TI)=0.9, T=600K")
print("\nRunning GPU calculation...")

# Force regeneration of the kernel with the fix
import os
cache_dir = os.path.expanduser('~/.cache/pycalphad_gpu')
if os.path.exists(cache_dir):
    print(f"Clearing GPU cache at {cache_dir}")
    import shutil
    shutil.rmtree(cache_dir)

result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
gpu_gm = result_gpu.GM.values.flatten()[0]

print(f"\nGPU GM: {gpu_gm:.6f} J/mol")

# Check actual composition
gpu_x_ti = result_gpu.X.sel(component='TI').values.flatten()
gpu_np = result_gpu.NP.values.flatten()

overall_x_ti = 0.0
for i, np_val in enumerate(gpu_np):
    if np_val > 1e-12:
        overall_x_ti += np_val * gpu_x_ti[i]

print(f"GPU X(TI): {overall_x_ti:.6f} (target: 0.900000)")
print(f"GPU residual: {abs(overall_x_ti - 0.9):.6f}")

# Compare with CPU
print("\nRunning CPU calculation...")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]

print(f"\nCPU GM: {cpu_gm:.6f} J/mol")
print(f"Error: {abs(cpu_gm - gpu_gm):.6f} J/mol")

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)

if abs(cpu_gm - gpu_gm) < 0.001:
    print("✓ SUCCESS! GPU now matches CPU within 0.001 J/mol")
    print("  The fix of treating all site fractions as independent works!")
else:
    print("✗ Still have error > 0.001 J/mol")
    print("  Need to investigate further")

if abs(overall_x_ti - 0.9) < 1e-6:
    print("✓ Composition constraint satisfied!")
else:
    print(f"✗ Composition constraint not satisfied: {overall_x_ti:.6f} vs 0.900000")