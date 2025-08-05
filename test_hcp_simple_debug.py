#!/usr/bin/env python
"""Simple debug of HCP issue comparing CPU vs GPU equilibrium matrices."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def simple_hcp_debug():
    """Simple debug of the HCP issue."""
    
    # Load database  
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # The failing case: includes both BCC_A2 and HCP_A3
    phases = ['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("Phase sublattice definitions:")
    for phase_name in phases:
        phase = dbf.phases[phase_name]
        print(f"  {phase_name}: sublattices = {phase.sublattices}")
    print()
    
    # CPU calculation - no verbose to avoid massive output
    print("=== CPU CALCULATION ===")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    
    print(f"CPU GM: {cpu_gm:.6f}")
    print("CPU active phases:")
    for phase, amount in zip(cpu_phases, cpu_np):
        if amount > 1e-8 and phase != '':
            print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation  
    print("\n=== GPU CALCULATION ===")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    
    print(f"GPU GM: {gpu_gm:.6f}")
    print("GPU active phases:")
    for phase, amount in zip(gpu_phases, gpu_np):
        if amount > 1e-8 and phase != '':
            print(f"  {phase}: {amount:.6f}")
    
    # Compare
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"\nGM difference: {gm_diff:.2e}")
    
    # Identify the key difference
    cpu_active = set(p for p, a in zip(cpu_phases, cpu_np) if a > 1e-8 and p != '')
    gpu_active = set(p for p, a in zip(gpu_phases, gpu_np) if a > 1e-8 and p != '')
    
    only_cpu = cpu_active - gpu_active
    only_gpu = gpu_active - cpu_active
    
    print(f"\nPhase differences:")
    if only_cpu:
        print(f"  Only in CPU: {only_cpu}")
    if only_gpu:
        print(f"  Only in GPU: {only_gpu}")
        
    if 'HCP_A3' in only_gpu:
        print(f"\n🔍 KEY ISSUE:")
        print(f"  HCP_A3 is incorrectly activated in GPU but not CPU")
        print(f"  HCP_A3 has sublattice (1.0, 0.5) - the 0.5 vacancy multiplier")
        print(f"  may be causing issues in GPU site fraction or normalization calculations")
        
        # Check if we have access to the site fractions
        print(f"\nTrying to analyze site fractions from result...")
        try:
            gpu_y = result_gpu.Y  # Site fractions
            print(f"GPU site fraction data available: {gpu_y.dims}")
        except:
            print("Site fraction data not available in output")

if __name__ == "__main__":
    simple_hcp_debug()