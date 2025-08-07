#!/usr/bin/env python
"""Test the newly identified failing conditions individually."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def test_failing_conditions():
    """Test the 4 failing conditions from the 1600-condition run."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Testing newly identified failing conditions individually...")
    print("=" * 70)
    
    # The 4 failing conditions from the 1600-condition test
    failing_conditions = [
        (0.8, 590),  # X(BI)=0.8, T=590K
        (0.7, 640),  # X(BI)=0.7, T=640K
        (0.7, 650),  # X(BI)=0.7, T=650K
        (0.6, 710),  # X(BI)=0.6, T=710K
    ]
    
    all_pass_individual = True
    
    for x_bi, temp in failing_conditions:
        print(f"\nCondition: X(BI)={x_bi}, T={temp}K")
        conditions = {v.X('BI'): x_bi, v.T: temp, v.P: 101325}
        
        # Test CPU
        cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False)
        cpu_gm = cpu_result.GM.values.item()
        
        # Test GPU
        gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True)
        gpu_gm = gpu_result.GM.values.item()
        
        diff = abs(cpu_gm - gpu_gm)
        status = "PASS" if diff < 1.0 else "FAIL"
        
        print(f"  CPU GM: {cpu_gm:.6f}")
        print(f"  GPU GM: {gpu_gm:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Status: {status}")
        
        if status == "FAIL":
            all_pass_individual = False
    
    print("\n" + "=" * 70)
    print("Testing all 4 conditions together in a batch...")
    
    # Test all 4 together
    conditions_batch = {
        v.X('BI'): [0.8, 0.7, 0.7, 0.6],
        v.T: [590, 640, 650, 710],
        v.P: 101325
    }
    
    cpu_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=False)
    gpu_batch = equilibrium(dbf, comps, phases, conditions_batch, gpu=True)
    
    cpu_gm_batch = cpu_batch.GM.values.flatten()
    gpu_gm_batch = gpu_batch.GM.values.flatten()
    
    print(f"\nBatch results (4 conditions):")
    batch_all_pass = True
    for i, (x_bi, temp) in enumerate(failing_conditions):
        if i < len(cpu_gm_batch):
            diff = abs(cpu_gm_batch[i] - gpu_gm_batch[i])
            status = "PASS" if diff < 1.0 else "FAIL"
            print(f"  X(BI)={x_bi}, T={temp}K: diff={diff:.6f} {status}")
            if status == "FAIL":
                batch_all_pass = False
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Individual tests: {'All PASS' if all_pass_individual else 'Some FAIL'}")
    print(f"  Batch test (4 conditions): {'All PASS' if batch_all_pass else 'Some FAIL'}")
    
    # Test with surrounding conditions to see if pattern emerges
    print("\n" + "=" * 70)
    print("Testing with surrounding conditions...")
    
    # Create a larger batch including neighboring conditions
    x_bi_expanded = [0.6, 0.7, 0.8]
    temp_expanded = [580, 590, 600, 640, 650, 660, 700, 710, 720]
    
    conditions_expanded = {
        v.X('BI'): x_bi_expanded,
        v.T: temp_expanded,
        v.P: 101325
    }
    
    cpu_expanded = equilibrium(dbf, comps, phases, conditions_expanded, gpu=False)
    gpu_expanded = equilibrium(dbf, comps, phases, conditions_expanded, gpu=True)
    
    cpu_gm_expanded = cpu_expanded.GM.values.flatten()
    gpu_gm_expanded = gpu_expanded.GM.values.flatten()
    
    print(f"\nExpanded batch results ({len(x_bi_expanded)} x {len(temp_expanded)} = {len(x_bi_expanded)*len(temp_expanded)} conditions):")
    
    failures = []
    for i in range(len(cpu_gm_expanded)):
        temp_idx = i // len(x_bi_expanded)
        x_idx = i % len(x_bi_expanded)
        
        if temp_idx < len(temp_expanded) and x_idx < len(x_bi_expanded):
            x_bi = x_bi_expanded[x_idx]
            temp = temp_expanded[temp_idx]
            
            diff = abs(cpu_gm_expanded[i] - gpu_gm_expanded[i])
            status = "PASS" if diff < 1.0 else "FAIL"
            
            if status == "FAIL":
                failures.append((x_bi, temp, i, diff))
                print(f"  [{i:2d}] X(BI)={x_bi}, T={temp}K: diff={diff:.6f} FAIL")
    
    if not failures:
        print("  All conditions PASS!")
    else:
        print(f"\nFailure analysis:")
        print(f"  Total failures: {len(failures)}")
        print(f"  Thread indices: {[f[2] for f in failures]}")

if __name__ == "__main__":
    test_failing_conditions()