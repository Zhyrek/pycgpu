#!/usr/bin/env python
"""Test GPU vs CPU with a large batch of conditions to verify memory fix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases
import time

def test_large_batch():
    """Test with 1600 conditions to verify memory allocation fix."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print(f"Testing large batch with {phases} phases...")
    
    # Create 1600 conditions (40x40 grid)
    x_bi_values = np.linspace(0.05, 0.95, 40)
    temp_values = np.linspace(350, 750, 40)
    
    conditions = {
        v.X('BI'): x_bi_values.tolist(),
        v.T: temp_values.tolist(),
        v.P: 101325
    }
    
    num_conditions = len(x_bi_values) * len(temp_values)
    print(f"Testing {num_conditions} conditions ({len(x_bi_values)} x {len(temp_values)} grid)")
    print("This will launch multiple GPU blocks to test memory allocation fix...")
    
    # CPU calculation
    print("\nRunning CPU calculation...")
    cpu_start = time.time()
    result_cpu = equilibrium(dbf, comps, phases, conditions, gpu=False)
    cpu_time = time.time() - cpu_start
    print(f"CPU completed in {cpu_time:.1f} seconds")
    
    # GPU calculation
    print("\nRunning GPU calculation...")
    gpu_start = time.time()
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True)
    gpu_time = time.time() - gpu_start
    print(f"GPU completed in {gpu_time:.1f} seconds")
    
    # Compare results
    cpu_gm = result_cpu.GM.values.flatten()
    gpu_gm = result_gpu.GM.values.flatten()
    
    failures = 0
    for i in range(len(cpu_gm)):
        if not np.isnan(cpu_gm[i]) and not np.isnan(gpu_gm[i]):
            diff = abs(cpu_gm[i] - gpu_gm[i])
            if diff > 1.0:
                failures += 1
    
    pass_rate = (len(cpu_gm) - failures) / len(cpu_gm) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Total conditions: {len(cpu_gm)}")
    print(f"  Passed: {len(cpu_gm) - failures}")
    print(f"  Failed: {failures}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  CPU time: {cpu_time:.1f} seconds")
    print(f"  GPU time: {gpu_time:.1f} seconds")
    print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"{'='*60}")
    
    # Success criteria
    if pass_rate >= 99.0:
        print("\n✓ TEST PASSED: Memory allocation fix is working!")
    else:
        print(f"\n✗ TEST FAILED: Pass rate {pass_rate:.1f}% is below 99%")

if __name__ == "__main__":
    test_large_batch()