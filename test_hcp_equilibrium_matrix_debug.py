#!/usr/bin/env python
"""Debug equilibrium matrix values for HCP_A3 phase with 0.5 vacancy site fraction."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def debug_hcp_matrix():
    """Debug the equilibrium matrix for the failing HCP case."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # Test the specific failing case: FCC_A1 + AU2BI_C15 + BCC_A2 + HCP_A3
    phases = ['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print(f"=== HCP EQUILIBRIUM MATRIX DEBUG ===")
    print(f"Phases: {phases}")
    print(f"Conditions: X(BI)=0.1, T=400K")
    
    # First check the phase definitions
    print(f"\nPhase sublattice definitions:")
    for phase_name in phases:
        phase = dbf.phases[phase_name]
        print(f"  {phase_name}: sublattices = {phase.sublattices}")
    
    print(f"\n=== CPU CALCULATION (VERBOSE) ===")
    try:
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        cpu_x = result_cpu.X.values[0,0,0,0]
        
        print(f"\nCPU FINAL RESULTS:")
        print(f"  GM: {cpu_gm:.6f}")
        print(f"  Active phases:")
        for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
                print(f"      Compositions: {cpu_x[i]}")
        
    except Exception as e:
        print(f"CPU ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== GPU CALCULATION (VERBOSE) ===")
    try:
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        gpu_x = result_gpu.X.values[0,0,0,0]
        
        print(f"\nGPU FINAL RESULTS:")
        print(f"  GM: {gpu_gm:.6f}")
        print(f"  Active phases:")
        for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
                print(f"      Compositions: {gpu_x[i]}")
        
    except Exception as e:
        print(f"GPU ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    try:
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"GM difference: {gm_diff:.2e}")
        
        print(f"\nPhase assemblage comparison:")
        print(f"CPU phases: {[p for p, a in zip(cpu_phases, cpu_np) if a > 1e-8 and p != '']}")
        print(f"GPU phases: {[p for p, a in zip(gpu_phases, gpu_np) if a > 1e-8 and p != '']}")
        
        if gm_diff > 1.0:
            print(f"\n❌ SIGNIFICANT DIFFERENCE DETECTED")
            print(f"This suggests the GPU is finding a different equilibrium")
            
            # Check which phases are different
            cpu_active = set(p for p, a in zip(cpu_phases, cpu_np) if a > 1e-8 and p != '')
            gpu_active = set(p for p, a in zip(gpu_phases, gpu_np) if a > 1e-8 and p != '')
            
            only_cpu = cpu_active - gpu_active
            only_gpu = gpu_active - cpu_active
            
            if only_cpu:
                print(f"Phases only in CPU result: {only_cpu}")
            if only_gpu:
                print(f"Phases only in GPU result: {only_gpu}")
                
            # Focus on HCP_A3 if it's active in GPU but not CPU
            if 'HCP_A3' in only_gpu:
                print(f"\n🔍 HCP_A3 ANALYSIS:")
                print(f"HCP_A3 is active in GPU but not CPU - this suggests")
                print(f"the GPU is incorrectly calculating the HCP_A3 equilibrium matrix")
                print(f"or site fractions due to the 0.5 vacancy sublattice multiplier")
                
        else:
            print(f"✅ Results match within tolerance")
            
    except Exception as e:
        print(f"Comparison error: {e}")

if __name__ == "__main__":
    debug_hcp_matrix()