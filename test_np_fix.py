#!/usr/bin/env python
"""Test if the NP normalization fix resolves the GM doubling issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the failing condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing NP normalization fix")
print("=" * 50)

# Run CPU calculation
print("\nCPU calculation:")
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_np = result_cpu.NP.values.flatten()
print(f"  GM = {cpu_gm:.2f} J/mol")
print(f"  NP values: {[f'{np:.6f}' for np in cpu_np if np > 1e-6]}")
print(f"  Sum of NP = {sum(cpu_np):.6f}")

# Run GPU calculation
print("\nGPU calculation:")
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_np = result_gpu.NP.values.flatten()
print(f"  GM = {gpu_gm:.2f} J/mol")
print(f"  NP values: {[f'{np:.6f}' for np in gpu_np if np > 1e-6]}")
print(f"  Sum of NP = {sum(gpu_np):.6f}")

# Compare results
print("\nComparison:")
print(f"  GM ratio (GPU/CPU): {gpu_gm/cpu_gm:.6f}")
print(f"  GM difference: {gpu_gm - cpu_gm:.2f} J/mol")
print(f"  NP sum difference: {sum(gpu_np) - sum(cpu_np):.6f}")

if abs(gpu_gm - cpu_gm) < 1.0:  # 1 J/mol tolerance
    print("\n✅ SUCCESS: GPU GM matches CPU within tolerance!")
else:
    print(f"\n❌ FAILED: GPU GM still differs by {abs(gpu_gm - cpu_gm):.2f} J/mol")