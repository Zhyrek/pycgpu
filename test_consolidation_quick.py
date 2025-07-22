#!/usr/bin/env python
"""Quick test to verify consolidation fix."""

from pycalphad import Database, equilibrium
import numpy as np

# Load TDB
tdb = Database('NbTi.tdb')
comps = ['NB', 'TI', 'VA']
phases = ['LIQUID', 'BCC_A2']

# Test a few conditions
test_conditions = [
    {'T': 1200, 'P': 101325, 'X(TI)': 0.9},
    {'T': 1000, 'P': 101325, 'X(TI)': 0.8},
    {'T': 900, 'P': 101325, 'X(TI)': 0.7},
]

print("Testing consolidation fix...")
print("="*60)

for cond in test_conditions:
    # Run CPU calculation
    eq_cpu = equilibrium(tdb, comps, phases, cond, gpu=False, verbose=False)
    cpu_np = eq_cpu.NP.values.flatten()
    cpu_stable = sum(1 for np in cpu_np if np > 1e-10)
    cpu_gm = eq_cpu.GM.values.item()
    
    # Run GPU calculation  
    eq_gpu = equilibrium(tdb, comps, phases, cond, gpu=True, verbose=False)
    gpu_np = eq_gpu.NP.values.flatten()
    gpu_stable = sum(1 for np in gpu_np if np > 1e-10)
    gpu_gm = eq_gpu.GM.values.item()
    
    # Compare
    gm_diff = abs(cpu_gm - gpu_gm)
    phases_match = cpu_stable == gpu_stable
    
    print(f"T={cond['T']}K, X(TI)={cond['X(TI)']}:")
    print(f"  CPU: {cpu_stable} phases, GM={cpu_gm:.1f}")
    print(f"  GPU: {gpu_stable} phases, GM={gpu_gm:.1f}")
    print(f"  GM diff: {gm_diff:.6f}, Phases match: {phases_match}")
    
    if not phases_match:
        print("  WARNING: Phase count mismatch!")
        
print("="*60)
print("Test complete.")