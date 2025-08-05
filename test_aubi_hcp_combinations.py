#!/usr/bin/env python
"""Test HCP_A3 with different phase combinations to isolate the issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_combination(phases, name):
    """Test a specific phase combination."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print(f"\n=== {name} ===")
    print(f"Phases: {phases}")
    
    try:
        # CPU
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        
        print(f"CPU GM: {cpu_gm:.6f}")
        print("CPU active phases:")
        for phase, amount in zip(cpu_phases, cpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
        # GPU
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        
        print(f"GPU GM: {gpu_gm:.6f}")
        print("GPU active phases:")
        for phase, amount in zip(gpu_phases, gpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
        
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"GM difference: {gm_diff:.2e}")
        
        if gm_diff < 1.0:
            print("✓ PASS")
            return True
        else:
            print("✗ FAIL")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

def run_hcp_combination_tests():
    """Test various combinations involving HCP_A3."""
    
    test_cases = [
        (['HCP_A3', 'LIQUID'], "HCP + LIQUID (known working)"),
        (['HCP_A3', 'FCC_A1'], "HCP + FCC (similar phases)"),
        (['HCP_A3', 'AU2BI_C15'], "HCP + C15"),
        (['HCP_A3', 'BCC_A2'], "HCP + BCC"),
        (['FCC_A1', 'HCP_A3', 'LIQUID'], "FCC + HCP + LIQUID"),
        (['AU2BI_C15', 'HCP_A3', 'LIQUID'], "C15 + HCP + LIQUID"),
        (['FCC_A1', 'HCP_A3', 'AU2BI_C15'], "FCC + HCP + C15 (problem case)"),
    ]
    
    results = []
    
    for phases, name in test_cases:
        success = test_combination(phases, name)
        results.append((name, success))
    
    print(f"\n=== SUMMARY ===")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

if __name__ == "__main__":
    run_hcp_combination_tests()