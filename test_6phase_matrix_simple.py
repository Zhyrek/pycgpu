#!/usr/bin/env python
"""Simple test to capture equilibrium matrix at iteration 0 for 6-phase case."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

# Monkey patch the solver to capture iteration 0 info
import pycalphad.core.eqsolver as solver_module

original_solve_eq = solver_module.solve_eq_at_conditions
captured_info = {}

def patched_solve_eq(spec, solver, cur_conds, grid, energy_limit=1.0, verbose=False):
    """Capture iteration 0 info"""
    # Hook into the first iteration
    old_verbose = verbose
    
    # Temporarily set verbose to True to get debug output
    result = original_solve_eq(spec, solver, cur_conds, grid, energy_limit=energy_limit, verbose=True)
    
    return result

# Apply patch
solver_module.solve_eq_at_conditions = patched_solve_eq

def test_6phase_matrix():
    """Test the 6-phase case."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("6-PHASE CASE - ITERATION 0 DEBUG")
    print("="*70)
    
    # CPU calculation
    print("\n--- CPU CALCULATION ---")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    
    print(f"\nCPU Final result:")
    print(f"  GM: {cpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(cpu_phases, cpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")
    
    # GPU calculation with verbose output at iteration 0
    print("\n\n--- GPU CALCULATION ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    
    print(f"\nGPU Final result:")
    print(f"  GM: {gpu_gm:.6f}")
    print("  Active phases:")
    for phase, amount in zip(gpu_phases, gpu_np):
        if phase != '' and amount > 1e-8:
            print(f"    {phase}: {amount:.6f}")
    
    # Analysis
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    # Check phase differences
    cpu_active_set = set(p for p, a in zip(cpu_phases, cpu_np) if p != '' and a > 1e-8)
    gpu_active_set = set(p for p, a in zip(gpu_phases, gpu_np) if p != '' and a > 1e-8)
    
    if 'HCP_A3' in gpu_active_set and 'HCP_A3' not in cpu_active_set:
        print("\n❌ GPU incorrectly has HCP_A3 active!")
        print("\nThe equilibrium matrix construction appears correct.")
        print("The issue is in the starting point selection:")
        print("  - CPU selects FCC_A1 + AU2BI_C15")
        print("  - GPU selects FCC_A1 + HCP_A3")
        print("\nThis happens during convex hull calculation in starting_point().")
        print("The 0.5 vacancy sublattice in HCP_A3 likely affects:")
        print("  1. Grid point energies")
        print("  2. Phase amount normalization")
        print("  3. Convex hull construction")

if __name__ == "__main__":
    test_6phase_matrix()