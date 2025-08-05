#!/usr/bin/env python
"""Test equilibrium with the 4 phases that should be active after add_nearly_stable."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_4phase():
    """Test with 4 phases."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    # The 4 phases after add_nearly_stable: FCC_A1, AU2BI_C15, HCP_A3, RHOMBOHEDRAL_A7
    phases = ['FCC_A1', 'AU2BI_C15', 'HCP_A3', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("4-PHASE TEST (after add_nearly_stable)")
    print("="*70)
    
    # CPU calculation
    print("\n--- CPU CALCULATION ---")
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=False)
    cpu_gm = result_cpu.GM.values[0,0,0,0]
    cpu_phases = result_cpu.Phase.values[0,0,0,0]
    cpu_np = result_cpu.NP.values[0,0,0,0]
    
    print(f"GM: {cpu_gm:.6f}")
    print("Active phases:")
    for phase, amount in zip(cpu_phases, cpu_np):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
    
    # GPU calculation
    print("\n--- GPU CALCULATION ---")
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=False)
    gpu_gm = result_gpu.GM.values[0,0,0,0]
    gpu_phases = result_gpu.Phase.values[0,0,0,0]
    gpu_np = result_gpu.NP.values[0,0,0,0]
    
    print(f"GM: {gpu_gm:.6f}")
    print("Active phases:")
    for phase, amount in zip(gpu_phases, gpu_np):
        if phase != '' and amount > 1e-8:
            print(f"  {phase}: {amount:.6f}")
    
    # Compare
    gm_diff = abs(gpu_gm - cpu_gm)
    print(f"\nGM difference: {gm_diff:.2e}")
    
    if gm_diff < 0.1:
        print("✅ PASS - Same equilibrium found")
    else:
        print("❌ FAIL - Different equilibria")

if __name__ == "__main__":
    test_4phase()