#!/usr/bin/env python
"""Trace single condition to find exact divergence point."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import os

# Enable detailed debug output
os.environ['CUDA_CACHE_DISABLE'] = '1'

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Single test condition
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Tracing single condition X(TI)=0.1, T=500K...")
print("="*80)

# Run CPU calculation with debug output
print("\nCPU Equilibrium:")
print("-"*80)
cpu_result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=True)

cpu_gm = float(cpu_result.GM.values)
cpu_phases = cpu_result.Phase.values[0, 0, 0, 0]
cpu_np = cpu_result.NP.values[0, 0, 0, 0]

print(f"\nCPU Result: GM={cpu_gm:.6f}")
active_cpu = [(i, cpu_phases[i], cpu_np[i]) for i in range(len(cpu_np)) if cpu_np[i] > 0]
for i, (idx, phase, amount) in enumerate(active_cpu):
    print(f"  Phase {i+1}: {phase}, amount={amount:.6f}")

# Run GPU calculation with debug output
print("\n" + "="*80)
print("\nGPU Equilibrium:")
print("-"*80)
gpu_result = equilibrium(db, components, phases, conditions, 
                        calc_opts={'pdens': 50}, verbose=True, gpu=True)

gpu_gm = float(gpu_result.GM.values)
gpu_phases = gpu_result.Phase.values[0, 0, 0, 0]
gpu_np = gpu_result.NP.values[0, 0, 0, 0]

print(f"\nGPU Result: GM={gpu_gm:.6f}")
active_gpu = [(i, gpu_phases[i], gpu_np[i]) for i in range(len(gpu_np)) if gpu_np[i] > 0]
for i, (idx, phase, amount) in enumerate(active_gpu):
    print(f"  Phase {i+1}: {phase}, amount={amount:.6f}")

print("\n" + "="*80)
print("Summary:")
print(f"  CPU: {len(active_cpu)} phases, GM={cpu_gm:.6f}")
print(f"  GPU: {len(active_gpu)} phases, GM={gpu_gm:.6f}")
print(f"  Difference: {abs(cpu_gm - gpu_gm):.6f}")
print("="*80)