#!/usr/bin/env python
"""Trace why GPU only detects 1 phase instead of 2 in miscibility gap."""

import numpy as np
from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.core.workspace import Workspace
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

# Run equilibrium calculations
print("\nRunning equilibrium calculations...")

# CPU equilibrium
print("\nCPU equilibrium:")
cpu_eq = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, verbose=False)
cpu_phases = cpu_eq.Phase.values[0][cpu_eq.NP.values[0] > 0]
cpu_amounts = cpu_eq.NP.values[0][cpu_eq.NP.values[0] > 0]
cpu_gm = float(cpu_eq.GM.values)

print(f"  Phases: {cpu_phases}")
print(f"  Amounts: {cpu_amounts}")
print(f"  GM: {cpu_gm:.6f}")

if len(cpu_phases) == 2 and all(p == 'BCC_A2' for p in cpu_phases):
    # Get compositions of each phase
    # Find indices of the two BCC_A2 phases
    phase_indices = np.where(cpu_eq.Phase.values[0] == 'BCC_A2')[0]
    
    # Get site fractions for Ti
    y_data = cpu_eq.Y.sel(vertex='BCC_A20TI').values[0]
    
    print(f"  Phase 1: Y(TI) = {y_data[phase_indices[0]]:.4f}")
    print(f"  Phase 2: Y(TI) = {y_data[phase_indices[1]]:.4f}")

# GPU equilibrium
print("\nGPU equilibrium:")
try:
    gpu_eq = equilibrium(db, components, phases, conditions, calc_opts={'pdens': 50}, verbose=False, gpu=True)
    gpu_phases = gpu_eq.Phase.values[0][gpu_eq.NP.values[0] > 0]
    gpu_amounts = gpu_eq.NP.values[0][gpu_eq.NP.values[0] > 0]
    gpu_gm = float(gpu_eq.GM.values)
    
    print(f"  Phases: {gpu_phases}")
    print(f"  Amounts: {gpu_amounts}")
    print(f"  GM: {gpu_gm:.6f}")
    
    if len(gpu_phases) == 1:
        # Get composition of single phase
        phase_idx = np.where(gpu_eq.Phase.values[0] == gpu_phases[0])[0][0]
        y_ti = gpu_eq.Y.sel(vertex='BCC_A20TI').values[0][phase_idx]
        print(f"  Single phase Y(TI) = {y_ti:.4f}")
        
        # Check if this is close to one of the CPU phase compositions
        if len(cpu_phases) == 2:
            y_data_cpu = cpu_eq.Y.sel(vertex='BCC_A20TI').values[0]
            phase_indices_cpu = np.where(cpu_eq.Phase.values[0] == 'BCC_A2')[0]
            y1_cpu = y_data_cpu[phase_indices_cpu[0]]
            y2_cpu = y_data_cpu[phase_indices_cpu[1]]
            
            print(f"\n  GPU Y(TI)={y_ti:.4f} vs CPU Y(TI)=[{y1_cpu:.4f}, {y2_cpu:.4f}]")
            
            if abs(y_ti - y1_cpu) < 0.01:
                print("  → GPU converged to Ti-poor phase")
            elif abs(y_ti - y2_cpu) < 0.01:
                print("  → GPU converged to Ti-rich phase")
            else:
                print("  → GPU converged to intermediate composition")
        
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Summary:")
print("- CPU correctly finds 2 BCC_A2 phases (miscibility gap)")
print("- GPU only finds 1 phase")
print("- This suggests the GPU solver may be:")
print("  1. Not initializing with both phases")
print("  2. Consolidating phases incorrectly")
print("  3. Having gradient/Hessian issues preventing phase separation")
print("="*80)