#!/usr/bin/env python
"""Test HCP issue with exact conditions from the failing multi-condition test."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_hcp_exact_conditions():
    """Test with exact conditions that were causing failures."""
    
    # Load database  
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    # Test different phase combinations
    test_cases = [
        (['FCC_A1', 'AU2BI_C15', 'BCC_A2', 'HCP_A3'], "Problem case"),
        (['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7'], "All phases"),
    ]
    
    # Test multiple individual conditions that should trigger the issue
    test_conditions = [
        ({v.X('BI'): 0.1, v.T: 400, v.P: 101325}, "X(BI)=0.1, T=400K"),
        ({v.X('BI'): 0.2, v.T: 400, v.P: 101325}, "X(BI)=0.2, T=400K"),
        ({v.X('BI'): 0.5, v.T: 600, v.P: 101325}, "X(BI)=0.5, T=600K"),
    ]
    
    for phases, phase_name in test_cases:
        print(f"\n=== Testing {phase_name} ===")
        print(f"Phases: {phases}")
        
        for conditions, cond_name in test_conditions:
            print(f"\n--- {cond_name} ---")
            
            try:
                # CPU
                result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
                cpu_gm = result_cpu.GM.values.flatten()[0]
                cpu_phases = result_cpu.Phase.values.flatten()
                cpu_np = result_cpu.NP.values.flatten()
                
                cpu_active = []
                for phase, amount in zip(cpu_phases, cpu_np):
                    if amount > 1e-8 and phase != '':
                        cpu_active.append(f"{phase}({amount:.3f})")
                
                # GPU
                result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
                gpu_gm = result_gpu.GM.values.flatten()[0]
                gpu_phases = result_gpu.Phase.values.flatten()
                gpu_np = result_gpu.NP.values.flatten()
                
                gpu_active = []
                for phase, amount in zip(gpu_phases, gpu_np):
                    if amount > 1e-8 and phase != '':
                        gpu_active.append(f"{phase}({amount:.3f})")
                
                gm_diff = abs(gpu_gm - cpu_gm)
                
                print(f"  CPU GM: {cpu_gm:.6f}, Active: {cpu_active}")
                print(f"  GPU GM: {gpu_gm:.6f}, Active: {gpu_active}")
                print(f"  Difference: {gm_diff:.2e}")
                
                if gm_diff > 1.0:
                    print(f"  ❌ FAIL - significant difference")
                    
                    cpu_phases_set = set(p.split('(')[0] for p in cpu_active)
                    gpu_phases_set = set(p.split('(')[0] for p in gpu_active)
                    
                    if 'HCP_A3' in gpu_phases_set and 'HCP_A3' not in cpu_phases_set:
                        print(f"  🔍 HCP_A3 incorrectly activated in GPU")
                elif gm_diff > 1e-6:
                    print(f"  ⚠️  Small difference (probably OK)")
                else:
                    print(f"  ✅ PASS")
                    
            except Exception as e:
                print(f"  💥 ERROR: {e}")
    
    # Now test multi-condition at once (batch processing)
    print(f"\n=== Multi-condition batch test ===")
    phases_all = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions_multi = {
        v.X('BI'): [0.1, 0.2, 0.5],  # Multiple conditions
        v.T: [400, 600],
        v.P: 101325
    }
    
    try:
        print("Running multi-condition CPU...")
        result_cpu = equilibrium(dbf, comps, phases_all, conditions_multi, gpu=False)
        cpu_gm = result_cpu.GM.values
        
        print("Running multi-condition GPU...")
        result_gpu = equilibrium(dbf, comps, phases_all, conditions_multi, gpu=True)
        gpu_gm = result_gpu.GM.values
        
        print(f"CPU result shape: {cpu_gm.shape}")
        print(f"GPU result shape: {gpu_gm.shape}")
        
        # Compare all conditions
        cpu_flat = cpu_gm.flatten()
        gpu_flat = gpu_gm.flatten()
        
        failures = 0
        total = min(len(cpu_flat), len(gpu_flat))
        
        for i in range(total):
            if not (np.isnan(cpu_flat[i]) or np.isnan(gpu_flat[i])):
                diff = abs(gpu_flat[i] - cpu_flat[i])
                if diff > 1.0:
                    failures += 1
        
        print(f"Multi-condition results: {failures}/{total} failures")
        
        if failures > 0:
            print(f"❌ Multi-condition processing triggers the HCP_A3 issue")
        else:
            print(f"✅ Multi-condition processing works correctly")
        
    except Exception as e:
        print(f"Multi-condition error: {e}")

if __name__ == "__main__":
    test_hcp_exact_conditions()