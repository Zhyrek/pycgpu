#!/usr/bin/env python
"""Debug GPU starting point selection for HCP_A3."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def debug_gpu_starting_point():
    """Debug why GPU selects HCP_A3 in starting point."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    # The failing case - all 6 phases
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    print("=== DEBUGGING GPU STARTING POINT ===")
    print(f"Phases: {phases}")
    print(f"Conditions: X(BI)=0.1, T=400K")
    
    # Check sublattices
    print("\nPhase sublattices:")
    for phase_name in phases:
        phase = dbf.phases[phase_name]
        print(f"  {phase_name}: {phase.sublattices}")
        
        # Calculate normalization factor
        site_ratio_sum = sum(phase.sublattices)
        print(f"    Site ratio sum: {site_ratio_sum}")
        
        # For HCP_A3 specifically
        if phase_name == 'HCP_A3':
            print(f"    ⚠️  HCP_A3 normalization factor: {site_ratio_sum} (should affect phase amounts)")
    
    # Run GPU calculation with verbose to see starting point details
    print("\n=== GPU CALCULATION (verbose) ===")
    try:
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
        
        # Extract results
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        
        print(f"\nGPU FINAL RESULTS:")
        print(f"  GM: {gpu_gm:.6f}")
        print("  Active phases:")
        for phase, amount in zip(gpu_phases, gpu_np):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
                
                # Check if this is HCP_A3
                if phase == 'HCP_A3':
                    print(f"      ❌ HCP_A3 is incorrectly active!")
                    print(f"      This likely means the starting point included HCP_A3")
                    print(f"      The 0.5 vacancy sublattice may affect:")
                    print(f"        - Grid point generation")
                    print(f"        - Phase amount normalization in convex hull")
                    print(f"        - Initial composition calculations")
                    
    except Exception as e:
        print(f"GPU ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare with CPU
    print("\n=== CPU CALCULATION ===")
    try:
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
        
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        
        print(f"CPU RESULTS:")
        print(f"  GM: {cpu_gm:.6f}")
        print("  Active phases:")
        for phase, amount in zip(cpu_phases, cpu_np):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
    except Exception as e:
        print(f"CPU ERROR: {e}")
    
    print("\n=== ANALYSIS ===")
    print("The issue is in the GPU starting point calculation.")
    print("When all 6 phases are present, the GPU incorrectly includes HCP_A3")
    print("in the starting point, while CPU correctly excludes it.")
    print("\nThe root cause is likely in how the GPU handles the 0.5 vacancy")
    print("sublattice during the convex hull calculation for the starting point.")

if __name__ == "__main__":
    debug_gpu_starting_point()