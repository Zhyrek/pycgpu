#!/usr/bin/env python
"""Final trace of GPU phase detection issue."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import cupy as cp
import os

# Clear cache
os.environ['CUDA_CACHE_DISABLE'] = '1'
cp.clear_memo()

# Load database
db = Database('NbTi.tdb')
components = ['NB', 'TI', 'VA']
phases = ['BCC_A2', 'HCP_A3']

# Test conditions - in miscibility gap
conditions = {
    v.T: 500,
    v.P: 101325,
    v.X('TI'): 0.1,
    v.N: 1
}

print("Tracing phase detection issue...")
print(f"Conditions: X(TI)={conditions[v.X('TI')]}, T={conditions[v.T]}K")
print("="*80)

# CPU equilibrium
print("\nCPU equilibrium:")
cpu_eq = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, verbose=False)

# Extract data with correct indexing
cpu_np = cpu_eq.NP.values[0, 0, 0, 0]  # Shape is (N, P, T, X_TI, vertex)
cpu_phases = cpu_eq.Phase.values[0, 0, 0, 0]
cpu_y = cpu_eq.Y.values[0, 0, 0, 0]  # Shape is (N, P, T, X_TI, vertex, component)
cpu_gm = cpu_eq.GM.values[0, 0, 0, 0]

# Find active phases
active_phases = []
for i in range(len(cpu_np)):
    if cpu_np[i] > 0:
        active_phases.append((i, cpu_phases[i], cpu_np[i], cpu_y[i]))

print(f"  GM: {cpu_gm:.6f}")
print(f"  Active phases: {len(active_phases)}")
for i, (idx, phase, amount, comp) in enumerate(active_phases):
    print(f"  Phase {i+1}: {phase}")
    print(f"    Amount: {amount:.6f}")
    print(f"    Y(NB): {comp[0]:.6f}, Y(TI): {comp[1]:.6f}")

# GPU equilibrium
print("\nGPU equilibrium:")
try:
    gpu_eq = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, verbose=False, gpu=True)
    
    # Extract data with correct indexing
    gpu_np = gpu_eq.NP.values[0, 0, 0, 0]
    gpu_phases = gpu_eq.Phase.values[0, 0, 0, 0]
    gpu_y = gpu_eq.Y.values[0, 0, 0, 0]
    gpu_gm = gpu_eq.GM.values[0, 0, 0, 0]
    
    # Find active phases
    gpu_active_phases = []
    for i in range(len(gpu_np)):
        if gpu_np[i] > 0:
            gpu_active_phases.append((i, gpu_phases[i], gpu_np[i], gpu_y[i]))
    
    print(f"  GM: {gpu_gm:.6f}")
    print(f"  Active phases: {len(gpu_active_phases)}")
    for i, (idx, phase, amount, comp) in enumerate(gpu_active_phases):
        print(f"  Phase {i+1}: {phase}")
        print(f"    Amount: {amount:.6f}")
        print(f"    Y(NB): {comp[0]:.6f}, Y(TI): {comp[1]:.6f}")
    
    # Compare
    print("\n" + "="*80)
    print("Comparison:")
    print(f"  CPU: {len(active_phases)} phases, GM = {cpu_gm:.6f}")
    print(f"  GPU: {len(gpu_active_phases)} phases, GM = {gpu_gm:.6f}")
    print(f"  GM difference: {abs(cpu_gm - gpu_gm):.6f}")
    
    if len(active_phases) == 2 and len(gpu_active_phases) == 1:
        print("\n  → GPU is missing phase separation!")
        print("  → This suggests the GPU minimizer may be:")
        print("     1. Not properly initializing both phases")
        print("     2. Incorrectly consolidating phases")
        print("     3. Having gradient/Hessian calculation issues")
        
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("="*80)