#!/usr/bin/env python
"""Test AuBi system with incremental phase addition to isolate problems."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def test_phase_combination(dbf, comps, phases, test_name):
    """Test a specific phase combination."""
    
    print(f"\n=== Testing {test_name} ===")
    print(f"Phases: {phases}")
    
    # Single condition test
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    try:
        # CPU calculation
        result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
        cpu_gm = result_cpu.GM.values[0,0,0,0]
        cpu_phases = result_cpu.Phase.values[0,0,0,0]
        cpu_np = result_cpu.NP.values[0,0,0,0]
        
        print(f"  CPU success: GM = {cpu_gm:.6f}")
        print(f"  CPU active phases:")
        for i, (phase, amount) in enumerate(zip(cpu_phases, cpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
        
        # GPU calculation
        result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_gm = result_gpu.GM.values[0,0,0,0]
        gpu_phases = result_gpu.Phase.values[0,0,0,0]
        gpu_np = result_gpu.NP.values[0,0,0,0]
        
        print(f"  GPU success: GM = {gpu_gm:.6f}")
        print(f"  GPU active phases:")
        for i, (phase, amount) in enumerate(zip(gpu_phases, gpu_np)):
            if amount > 1e-8 and phase != '':
                print(f"    {phase}: {amount:.6f}")
        
        # Compare
        gm_diff = abs(gpu_gm - cpu_gm)
        print(f"  GM difference: {gm_diff:.2e}")
        
        if gm_diff < 1.0:
            print(f"  ✓ PASS")
            return True
        else:
            print(f"  ✗ FAIL (difference too large)")
            return False
            
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return False

def run_incremental_test():
    """Run incremental phase testing."""
    
    # Load database and get all phases
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    all_phases = filter_phases(dbf, comps)
    
    print(f"All available phases: {all_phases}")
    
    # Test progressively more phases
    phase_combinations = [
        (['FCC_A1', 'LIQUID'], "Two phases (known working)"),
        (['FCC_A1', 'LIQUID', 'AU2BI_C15'], "Add C15"),
        (['FCC_A1', 'LIQUID', 'AU2BI_C15', 'BCC_A2'], "Add BCC_A2"),
        (['FCC_A1', 'LIQUID', 'AU2BI_C15', 'BCC_A2', 'HCP_A3'], "Add HCP_A3"),
        (['FCC_A1', 'LIQUID', 'AU2BI_C15', 'BCC_A2', 'HCP_A3', 'RHOMBOHEDRAL_A7'], "All phases"),
        (all_phases, "All phases (filter_phases)")
    ]
    
    results = []
    
    for phases, description in phase_combinations:
        success = test_phase_combination(dbf, comps, phases, description)
        results.append((description, success))
    
    print(f"\n=== SUMMARY ===")
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} {description}")

if __name__ == "__main__":
    run_incremental_test()