#!/usr/bin/env python3
"""
Simple comparison to understand the fundamental difference between CPU and GPU
"""

import numpy as np
import pycalphad as pyc
from pycalphad import Database, equilibrium, variables as v
import os
import glob

def clear_cupy_kernel_cache():
    cache_dir = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.exists(cache_dir):
        cubin_files = glob.glob(os.path.join(cache_dir, "*.cubin"))
        for cubin_file in cubin_files:
            try:
                os.remove(cubin_file)
            except OSError:
                pass

def compare_basic_results():
    print("=== SIMPLE CPU vs GPU COMPARISON ===")
    clear_cupy_kernel_cache()
    
    tdb = Database("NbTi.tdb")
    phases = ["BCC_A2"]
    comps = ["NB", "TI", "VA"]
    
    conditions = {
        v.X("TI"): 0.1,
        v.T: 800,
        v.P: 101325
    }
    
    print("Testing single-phase system:")
    print(f"  Phases: {phases}")
    print(f"  Conditions: {conditions}")
    print(f"  Expected: Single BCC_A2 phase with ~90% NB, 10% TI")
    
    # CPU
    print("\n--- CPU Result ---")
    cpu_result = equilibrium(tdb, comps, phases, conditions, gpu=False, verbose=False)
    
    cpu_gm = cpu_result.GM.values.flatten()[0]
    cpu_phases = cpu_result.Phase.values.flatten()
    cpu_np = cpu_result.NP.values.flatten()
    cpu_x = cpu_result.X.values
    
    # Count actual active phases in CPU
    cpu_active_mask = ~np.isnan(cpu_np) & (cpu_np > 1e-10)
    cpu_active_phases = np.sum(cpu_active_mask)
    
    print(f"GM: {cpu_gm:.2f} J/mol")
    print(f"Active phases: {cpu_active_phases}")
    print(f"Phase names: {cpu_phases[cpu_active_mask]}")
    print(f"Phase amounts: {cpu_np[cpu_active_mask]}")
    
    # Extract composition of active phases
    for i, is_active in enumerate(cpu_active_mask):
        if is_active and i < cpu_x.shape[-2]:
            comp = cpu_x.reshape(-1, cpu_x.shape[-1])[i][:2]  # NB, TI only
            print(f"Phase {i} composition: NB={comp[0]:.3f}, TI={comp[1]:.3f}")
    
    # GPU
    print("\n--- GPU Result ---")
    gpu_result = equilibrium(tdb, comps, phases, conditions, gpu=True, verbose=True)
    
    gpu_gm = gpu_result.GM.values.flatten()[0]
    gpu_phases = gpu_result.Phase.values.flatten()
    gpu_np = gpu_result.NP.values.flatten()
    gpu_x = gpu_result.X.values
    
    # Count actual active phases in GPU
    gpu_active_mask = gpu_np > 1e-10
    gpu_active_phases = np.sum(gpu_active_mask)
    
    print(f"GM: {gpu_gm:.2f} J/mol")
    print(f"Active phases: {gpu_active_phases}")
    print(f"Phase names: {gpu_phases[gpu_active_mask]}")
    print(f"Phase amounts: {gpu_np[gpu_active_mask]}")
    
    # Extract composition of active phases
    for i, is_active in enumerate(gpu_active_mask):
        if is_active and i < gpu_x.shape[-2]:
            comp = gpu_x.reshape(-1, gpu_x.shape[-1])[i][:2]  # NB, TI only
            print(f"Phase {i} composition: NB={comp[0]:.3f}, TI={comp[1]:.3f}")
    
    # Analysis
    print("\n--- Analysis ---")
    gm_diff = abs(cpu_gm - gpu_gm)
    print(f"GM difference: {gm_diff:.2f} J/mol")
    print(f"Phase count: CPU={cpu_active_phases}, GPU={gpu_active_phases}")
    
    if cpu_active_phases == 1 and gpu_active_phases == 2:
        print("\n🔍 ROOT CAUSE IDENTIFIED:")
        print("   CPU finds 1 phase equilibrium (correct)")
        print("   GPU finds 2 phase equilibrium (incorrect)")
        print("   This is a PHASE STABILITY ALGORITHM issue, not just numerical precision")
        print("\n   The GPU minimization is converging to a two-phase tie-line")
        print("   instead of the single-phase solution that's thermodynamically correct.")
        print("\n   Possible causes:")
        print("   1. Different initial conditions leading to wrong local minimum")
        print("   2. Incomplete phase removal/addition algorithm")
        print("   3. Different convergence criteria")
        print("   4. Bug in GPU energy/chemical potential calculation")
    
    return cpu_active_phases, gpu_active_phases, gm_diff

if __name__ == "__main__":
    compare_basic_results()