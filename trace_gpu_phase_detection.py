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

# First run calculate to see grid points
print("\nRunning calculate to check grid points...")
calc_result = calculate(db, components, phases, T=conditions[v.T], P=conditions[v.P], 
                        output='GM', model=None, points={'BCC_A2': 50, 'HCP_A3': 50})

# Check BCC_A2 grid
bcc_mask = calc_result.Phase == 'BCC_A2'
bcc_gm = calc_result.GM.values[bcc_mask]
bcc_x_ti = calc_result.X.sel(component='TI').values[bcc_mask]

print(f"\nBCC_A2 grid points: {len(bcc_gm)}")
print("Sample BCC_A2 compositions and energies:")
for i in range(0, len(bcc_gm), 10):
    print(f"  X(TI)={bcc_x_ti[i]:.4f}, GM={bcc_gm[i]:.1f}")

# Find compositions near X(TI)=0.1
near_01 = np.abs(bcc_x_ti - 0.1) < 0.05
if np.any(near_01):
    print(f"\nBCC_A2 near X(TI)=0.1:")
    for x, gm in zip(bcc_x_ti[near_01], bcc_gm[near_01]):
        print(f"  X(TI)={x:.4f}, GM={gm:.1f}")

# Check for miscibility gap in BCC_A2
print("\nChecking for miscibility gap in BCC_A2...")
# Sort by composition
sort_idx = np.argsort(bcc_x_ti)
x_sorted = bcc_x_ti[sort_idx]
gm_sorted = bcc_gm[sort_idx]

# Look for non-convex regions
d2gm_dx2 = np.diff(np.diff(gm_sorted) / np.diff(x_sorted)) / np.diff(x_sorted[:-1])
negative_d2gm = d2gm_dx2 < 0

if np.any(negative_d2gm):
    print("✓ Found negative second derivative (miscibility gap)")
    gap_start = np.where(negative_d2gm)[0][0]
    gap_end = np.where(negative_d2gm)[0][-1] + 2
    print(f"  Gap region: X(TI) = {x_sorted[gap_start]:.3f} to {x_sorted[gap_end]:.3f}")
else:
    print("✗ No negative second derivative found")

# Now check equilibrium calculations
print("\n" + "="*80)
print("Running equilibrium calculations...")

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
    cpu_y = cpu_eq.Y.values[0]
    idx1 = np.where(cpu_eq.Phase.values[0] == 'BCC_A2')[0][0]
    idx2 = np.where(cpu_eq.Phase.values[0] == 'BCC_A2')[0][1]
    
    x_ti_1 = cpu_y[idx1, cpu_eq.component_list.index('TI')]
    x_ti_2 = cpu_y[idx2, cpu_eq.component_list.index('TI')]
    
    print(f"  Phase 1: X(TI) = {x_ti_1:.4f}")
    print(f"  Phase 2: X(TI) = {x_ti_2:.4f}")

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
        gpu_y = gpu_eq.Y.values[0]
        idx = np.where(gpu_eq.Phase.values[0] == gpu_phases[0])[0][0]
        x_ti = gpu_y[idx, gpu_eq.component_list.index('TI')]
        print(f"  Single phase X(TI) = {x_ti:.4f}")
        
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("Summary:")
print("- CPU correctly finds 2 BCC_A2 phases (miscibility gap)")
print("- GPU only finds 1 phase")
print("- This suggests the GPU solver may be:")
print("  1. Not detecting the phase separation")
print("  2. Consolidating phases incorrectly")
print("  3. Having issues with the gradient calculations")
print("="*80)