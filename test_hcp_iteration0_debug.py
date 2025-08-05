#!/usr/bin/env python
"""Debug equilibrium solver at iteration 0 for 4-phase vs 6-phase cases."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import unpack_kwarg
import warnings
warnings.filterwarnings("ignore")

def debug_iteration0():
    """Compare iteration 0 behavior between 4-phase and 6-phase cases."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    # Test cases
    test_cases = [
        (['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3'], "4-phase case"),
        (['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7'], "6-phase case (all)"),
    ]
    
    for phases, case_name in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing {case_name}")
        print(f"Phases: {phases}")
        print(f"{'='*80}")
        
        # Check phase sublattices
        print(f"\nPhase sublattice definitions:")
        for phase_name in phases:
            if phase_name in dbf.phases:
                phase = dbf.phases[phase_name]
                print(f"  {phase_name}: sublattices = {phase.sublattices}")
        
        # Run with verbose to see starting point info
        print(f"\n--- Starting Point Calculation ---")
        
        # CPU calculation with verbose (this will show starting point details)
        print(f"\nCPU Calculation:")
        try:
            # Use calc_opts to potentially see more debug info
            calc_opts = {'pdens': 60}  # Lower density to see grid points more clearly
            
            result_cpu = equilibrium(dbf, comps, phases, conditions, 
                                   gpu=False, verbose=True, calc_opts=calc_opts)
            cpu_gm = result_cpu.GM.values[0,0,0,0]
            cpu_phases = result_cpu.Phase.values[0,0,0,0]
            cpu_np = result_cpu.NP.values[0,0,0,0]
            cpu_mu = result_cpu.MU.values[0,0,0,0]
            
            print(f"\nCPU RESULTS:")
            print(f"  GM: {cpu_gm:.6f}")
            print(f"  MU: {cpu_mu}")
            print("  Active phases:")
            for phase, amount in zip(cpu_phases, cpu_np):
                if amount > 1e-8 and phase != '':
                    print(f"    {phase}: {amount:.6f}")
        except Exception as e:
            print(f"CPU ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # GPU calculation
        print(f"\n\nGPU Calculation:")
        try:
            result_gpu = equilibrium(dbf, comps, phases, conditions, 
                                   gpu=True, verbose=True, calc_opts=calc_opts)
            gpu_gm = result_gpu.GM.values[0,0,0,0]
            gpu_phases = result_gpu.Phase.values[0,0,0,0]
            gpu_np = result_gpu.NP.values[0,0,0,0]
            gpu_mu = result_gpu.MU.values[0,0,0,0]
            
            print(f"\nGPU RESULTS:")
            print(f"  GM: {gpu_gm:.6f}")
            print(f"  MU: {gpu_mu}")
            print("  Active phases:")
            for phase, amount in zip(gpu_phases, gpu_np):
                if amount > 1e-8 and phase != '':
                    print(f"    {phase}: {amount:.6f}")
        except Exception as e:
            print(f"GPU ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Compare
        try:
            gm_diff = abs(gpu_gm - cpu_gm)
            mu_diff = np.max(np.abs(gpu_mu - cpu_mu))
            
            print(f"\n--- Comparison ---")
            print(f"GM difference: {gm_diff:.2e}")
            print(f"Max MU difference: {mu_diff:.2e}")
            
            if gm_diff > 1.0:
                print("❌ FAIL - Different equilibria found")
                
                # Analyze differences
                cpu_active = set(p for p, a in zip(cpu_phases, cpu_np) if a > 1e-8 and p != '')
                gpu_active = set(p for p, a in zip(gpu_phases, gpu_np) if a > 1e-8 and p != '')
                
                if cpu_active != gpu_active:
                    print(f"\nPhase assemblage mismatch:")
                    print(f"  CPU phases: {cpu_active}")
                    print(f"  GPU phases: {gpu_active}")
                    
                    only_gpu = gpu_active - cpu_active
                    if 'HCP_A3' in only_gpu:
                        print(f"\n🔍 KEY FINDING:")
                        print(f"  HCP_A3 is active in GPU but not CPU")
                        print(f"  HCP_A3 has sublattice (1.0, 0.5)")
                        print(f"  The 0.5 vacancy multiplier may affect:")
                        print(f"    - Starting point grid generation")
                        print(f"    - Phase amount normalization") 
                        print(f"    - Site fraction calculations")
                        print(f"    - Equilibrium matrix construction")
            else:
                print("✅ PASS - Same equilibrium found")
                
        except Exception as e:
            print(f"Comparison error: {e}")

if __name__ == "__main__":
    debug_iteration0()