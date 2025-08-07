#!/usr/bin/env python
"""Test specific failing conditions individually vs in batch."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def test_individual_conditions():
    """Test failing conditions individually."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Testing failing conditions individually...")
    print("=" * 60)
    
    # Test condition 1: X(BI)=0.3, T=500K
    print("\nCondition 1: X(BI)=0.3, T=500K")
    conditions1 = {v.X('BI'): 0.3, v.T: 500, v.P: 101325}
    
    cpu1 = equilibrium(dbf, comps, phases, conditions1, gpu=False)
    gpu1 = equilibrium(dbf, comps, phases, conditions1, gpu=True)
    
    cpu_gm1 = cpu1.GM.values.item()
    gpu_gm1 = gpu1.GM.values.item()
    diff1 = abs(cpu_gm1 - gpu_gm1)
    
    print(f"  CPU GM: {cpu_gm1:.6f}")
    print(f"  GPU GM: {gpu_gm1:.6f}")
    print(f"  Difference: {diff1:.6f}")
    print(f"  Status: {'PASS' if diff1 < 1.0 else 'FAIL'}")
    
    # Test condition 2: X(BI)=0.2, T=600K
    print("\nCondition 2: X(BI)=0.2, T=600K")
    conditions2 = {v.X('BI'): 0.2, v.T: 600, v.P: 101325}
    
    cpu2 = equilibrium(dbf, comps, phases, conditions2, gpu=False)
    gpu2 = equilibrium(dbf, comps, phases, conditions2, gpu=True)
    
    cpu_gm2 = cpu2.GM.values.item()
    gpu_gm2 = gpu2.GM.values.item()
    diff2 = abs(cpu_gm2 - gpu_gm2)
    
    print(f"  CPU GM: {cpu_gm2:.6f}")
    print(f"  GPU GM: {gpu_gm2:.6f}")
    print(f"  Difference: {diff2:.6f}")
    print(f"  Status: {'PASS' if diff2 < 1.0 else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("Testing both conditions in a batch...")
    
    # Test both conditions together
    conditions_batch = {
        v.X('BI'): [0.3, 0.2],
        v.T: [500, 600],
        v.P: 101325
    }
    
    cpu_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=False)
    gpu_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=True)
    
    cpu_gm_batch = cpu_batch.GM.values.flatten()
    gpu_gm_batch = gpu_batch.GM.values.flatten()
    
    print(f"\nBatch results:")
    for i, (x_bi, temp) in enumerate([(0.3, 500), (0.2, 600)]):
        diff = abs(cpu_gm_batch[i] - gpu_gm_batch[i])
        print(f"  X(BI)={x_bi}, T={temp}K:")
        print(f"    CPU GM: {cpu_gm_batch[i]:.6f}")
        print(f"    GPU GM: {gpu_gm_batch[i]:.6f}")
        print(f"    Difference: {diff:.6f}")
        print(f"    Status: {'PASS' if diff < 1.0 else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("Testing in larger batch with surrounding conditions...")
    
    # Test in context with surrounding conditions
    conditions_large = {
        v.X('BI'): [0.2, 0.3, 0.4],
        v.T: [400, 500, 600],
        v.P: 101325
    }
    
    cpu_large = equilibrium(dbf, comps, phases, conditions_large, gpu=False)
    gpu_large = equilibrium(dbf, comps, phases, conditions_large, gpu=True)
    
    cpu_gm_large = cpu_large.GM.values.flatten()
    gpu_gm_large = gpu_large.GM.values.flatten()
    
    # Map out which indices correspond to our conditions of interest
    print(f"\nLarge batch results (9 conditions):")
    idx = 0
    for t in [400, 500, 600]:
        for x in [0.2, 0.3, 0.4]:
            diff = abs(cpu_gm_large[idx] - gpu_gm_large[idx])
            status = 'PASS' if diff < 1.0 else 'FAIL'
            marker = " <--" if (x == 0.3 and t == 500) or (x == 0.2 and t == 600) else ""
            print(f"  [{idx}] X(BI)={x}, T={t}K: diff={diff:.6f} {status}{marker}")
            idx += 1

if __name__ == "__main__":
    test_individual_conditions()