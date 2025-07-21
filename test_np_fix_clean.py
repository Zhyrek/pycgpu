#!/usr/bin/env python
"""Test if the NP normalization fix resolves the GM doubling issue - clean version."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import warnings
warnings.filterwarnings("ignore")

# Load database and set up calculation
dbf = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = filter_phases(dbf, comps)

# Test the failing condition: X(TI)=0.1, T=600K
conditions = {v.X('TI'): 0.1, v.T: 600, v.P: 101325}

print("Testing NP normalization fix (clean output)")
print("=" * 60)

# Run CPU calculation
result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
cpu_gm = result_cpu.GM.values.flatten()[0]
cpu_np = result_cpu.NP.values.flatten()
cpu_phases = result_cpu.Phase.values.flatten()

print("\nCPU calculation:")
print(f"  GM = {cpu_gm:.2f} J/mol")
active_cpu = [(phase, np) for phase, np in zip(cpu_phases, cpu_np) if np > 1e-6]
for phase, np in active_cpu:
    print(f"  {phase}: NP = {np:.6f}")
print(f"  Total NP = {sum(np for _, np in active_cpu):.6f}")

# Run GPU calculation
result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
gpu_gm = result_gpu.GM.values.flatten()[0]
gpu_np = result_gpu.NP.values.flatten()
gpu_phases = result_gpu.Phase.values.flatten()

print("\nGPU calculation:")
print(f"  GM = {gpu_gm:.2f} J/mol")
active_gpu = [(phase, np) for phase, np in zip(gpu_phases, gpu_np) if np > 1e-6]
for phase, np in active_gpu:
    print(f"  {phase}: NP = {np:.6f}")
print(f"  Total NP = {sum(np for _, np in active_gpu):.6f}")

# Compare results
print("\nComparison:")
print(f"  GM ratio (GPU/CPU): {gpu_gm/cpu_gm:.6f}")
print(f"  GM difference: {gpu_gm - cpu_gm:.2f} J/mol")
print(f"  CPU phases: {len(active_cpu)}, GPU phases: {len(active_gpu)}")

if abs(gpu_gm - cpu_gm) < 10.0:  # 10 J/mol tolerance
    print("\n✅ SUCCESS: GPU GM matches CPU within tolerance!")
    print("✅ NP normalization fix is working!")
else:
    print(f"\n❌ FAILED: GPU GM still differs by {abs(gpu_gm - cpu_gm):.2f} J/mol")

# Test more conditions
print("\n" + "="*60)
print("Testing additional conditions:")
test_conditions = [
    {v.X('TI'): 0.5, v.T: 600, v.P: 101325},
    {v.X('TI'): 0.9, v.T: 600, v.P: 101325},
]

for cond in test_conditions:
    x_ti = cond[v.X('TI')]
    
    result_cpu = equilibrium(dbf, comps, phases, cond, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values.flatten()[0]
    
    result_gpu = equilibrium(dbf, comps, phases, cond, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values.flatten()[0]
    
    diff = gpu_gm - cpu_gm
    ratio = gpu_gm / cpu_gm
    status = "✅" if abs(diff) < 10.0 else "❌"
    
    print(f"\nX(TI)={x_ti}: CPU GM={cpu_gm:.2f}, GPU GM={gpu_gm:.2f}")
    print(f"  Difference: {diff:.2f} J/mol, Ratio: {ratio:.6f} {status}")