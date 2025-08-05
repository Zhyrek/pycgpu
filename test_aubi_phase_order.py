#!/usr/bin/env python
"""Test AuBi system with different phase orderings to isolate the issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_phase_order(phases, name):
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
        cpu_active = []
        for phase, amount in zip(cpu_phases, cpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
                cpu_active.append(f"{phase}({amount:.3f})")
        
        # GPU
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        
        print(f"GPU GM: {gpu_gm:.6f}")
        print("GPU active phases:")
        gpu_active = []
        for phase, amount in zip(gpu_phases, gpu_np):
            if amount > 1e-8 and phase != '':
                print(f"  {phase}: {amount:.6f}")
                gpu_active.append(f"{phase}({amount:.3f})")
        
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"GM difference: {gm_diff:.2e}")
        
        # Check if phase assemblages match
        phases_match = set(cpu_active) == set(gpu_active)
        print(f"Phase assemblages match: {phases_match}")
        
        if gm_diff < 1.0:
            print("✓ PASS")
            return True, cpu_active, gpu_active
        else:
            print("✗ FAIL")
            return False, cpu_active, gpu_active
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, [], []

def run_phase_order_tests():
    """Test different phase orderings."""
    
    # Test systematic phase addition  
    base_phases = ['FCC_A1', 'LIQUID', 'AU2BI_C15']
    additional_phases = ['BCC_A2', 'HCP_A3', 'RHOMBOHEDRAL_A7']
    
    test_cases = [
        (base_phases, "Base: FCC + LIQUID + C15"),
        (base_phases + ['BCC_A2'], "Add BCC_A2"),
        (base_phases + ['HCP_A3'], "Add HCP_A3"),
        (base_phases + ['RHOMBOHEDRAL_A7'], "Add RHOMBOHEDRAL_A7"),
        (base_phases + ['BCC_A2', 'HCP_A3'], "Add BCC_A2 + HCP_A3"),
        (base_phases + ['BCC_A2', 'RHOMBOHEDRAL_A7'], "Add BCC_A2 + RHOMBOHEDRAL_A7"),
        (base_phases + ['HCP_A3', 'RHOMBOHEDRAL_A7'], "Add HCP_A3 + RHOMBOHEDRAL_A7"),
        (base_phases + additional_phases, "All phases"),
    ]
    
    # Also test different orderings of the same phases
    all_phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    test_cases.extend([
        (all_phases, "All phases (different order)"),
        (['LIQUID', 'FCC_A1', 'HCP_A3', 'AU2BI_C15', 'BCC_A2', 'RHOMBOHEDRAL_A7'], "All phases (LIQUID first)"),
        (['HCP_A3', 'FCC_A1', 'LIQUID', 'AU2BI_C15', 'BCC_A2', 'RHOMBOHEDRAL_A7'], "All phases (HCP_A3 first)"),
    ])
    
    results = []
    
    for phases, name in test_cases:
        success, cpu_active, gpu_active = test_phase_order(phases, name)
        results.append((name, success, cpu_active, gpu_active))
    
    print(f"\n=== SUMMARY ===")
    for name, success, cpu_active, gpu_active in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
        if not success:
            print(f"    CPU: {', '.join(cpu_active)}")
            print(f"    GPU: {', '.join(gpu_active)}")

if __name__ == "__main__":
    run_phase_order_tests()