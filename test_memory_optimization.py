#!/usr/bin/env python
"""Test that memory optimization with cp.empty works correctly."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time
import cupy as cp

def test_memory_optimized_gpu():
    """Test GPU equilibrium with optimized memory allocation."""
    
    print("Testing Memory-Optimized GPU Equilibrium")
    print("=" * 60)
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'RHOMBOHEDRAL_A7']
    
    # Memory pool for tracking
    mempool = cp.get_default_memory_pool()
    
    # Test with different condition sizes
    test_cases = [
        (10, 10, "Small (100 conditions)"),
        (30, 30, "Medium (900 conditions)"),
        (50, 50, "Large (2500 conditions)"),
    ]
    
    for n_comp, n_temp, desc in test_cases:
        print(f"\n{desc}:")
        print("-" * 40)
        
        # Create conditions
        x_bi_values = np.linspace(0.05, 0.95, n_comp)
        temp_values = np.linspace(400, 1200, n_temp)
        
        # Create proper multi-condition grid
        conditions = {}
        conditions[v.T] = np.repeat(temp_values, n_comp)
        conditions[v.P] = 101325 * np.ones(n_temp * n_comp)
        conditions[v.X('BI')] = np.tile(x_bi_values, n_temp)
        
        # Clear memory
        mempool.free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        initial_mem = mempool.used_bytes() / (1024**2)
        
        # Run GPU calculation with timing
        start_time = time.time()
        try:
            result = equilibrium(dbf, comps, phases, conditions, 
                               gpu=True, verbose=False,
                               calc_opts={'pdens': 50})
            gpu_time = time.time() - start_time
            
            # Check results
            gm_values = result.GM.values.flatten()
            valid = np.sum(~np.isnan(gm_values))
            peak_mem = mempool.used_bytes() / (1024**2)
            
            print(f"  Conditions: {n_comp * n_temp}")
            print(f"  Valid results: {valid}/{len(gm_values)} ({100*valid/len(gm_values):.1f}%)")
            print(f"  Time: {gpu_time:.3f} seconds")
            print(f"  Peak memory: {peak_mem:.1f} MB")
            print(f"  Memory allocated: {peak_mem - initial_mem:.1f} MB")
            
            # Verify results are reasonable
            valid_gm = gm_values[~np.isnan(gm_values)]
            if len(valid_gm) > 0:
                print(f"  GM range: [{np.min(valid_gm):.1f}, {np.max(valid_gm):.1f}] J/mol")
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
    
    print("\n" + "=" * 60)
    print("Memory optimization test complete!")
    print("Using cp.empty instead of cp.zeros should show:")
    print("  - Faster allocation times")
    print("  - Same memory usage")
    print("  - Identical results")

def compare_allocation_times():
    """Direct comparison of allocation times."""
    
    print("\n" + "=" * 60)
    print("Direct Allocation Time Comparison")
    print("=" * 60)
    
    # Simulate allocation for 2500 threads
    num_threads = 2500
    
    # Test zeros allocation
    start = time.time()
    arrays_zeros = {
        'A': cp.zeros((num_threads, 138*138), dtype=cp.float64),
        'U': cp.zeros((num_threads, 138*138), dtype=cp.float64),
        'V': cp.zeros((num_threads, 138*138), dtype=cp.float64),
        'hess': cp.zeros((num_threads, 100*100), dtype=cp.float64),
    }
    cp.cuda.runtime.deviceSynchronize()
    zeros_time = time.time() - start
    
    del arrays_zeros
    cp.get_default_memory_pool().free_all_blocks()
    
    # Test empty allocation
    start = time.time()
    arrays_empty = {
        'A': cp.empty((num_threads, 138*138), dtype=cp.float64),
        'U': cp.empty((num_threads, 138*138), dtype=cp.float64),
        'V': cp.empty((num_threads, 138*138), dtype=cp.float64),
        'hess': cp.empty((num_threads, 100*100), dtype=cp.float64),
    }
    cp.cuda.runtime.deviceSynchronize()
    empty_time = time.time() - start
    
    print(f"Allocating work arrays for {num_threads} threads:")
    print(f"  cp.zeros: {zeros_time:.4f} seconds")
    print(f"  cp.empty: {empty_time:.4f} seconds")
    print(f"  Speedup: {zeros_time/empty_time:.2f}x")
    
    del arrays_empty

if __name__ == "__main__":
    test_memory_optimized_gpu()
    compare_allocation_times()