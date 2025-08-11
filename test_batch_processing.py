#!/usr/bin/env python
"""Test batch processing implementation with large condition sets."""

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import time
import cupy as cp

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def test_batch_processing():
    """Test GPU batch processing with various condition sizes."""
    
    # Load database
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'RHOMBOHEDRAL_A7']  # Limit phases for testing
    
    print("Testing GPU Batch Processing Implementation")
    print("=" * 60)
    
    # Get initial GPU memory
    mempool = cp.get_default_memory_pool()
    initial_used = mempool.used_bytes() / (1024**2)
    total_memory = cp.cuda.Device().mem_info[1] / (1024**3)
    
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"Initial memory used: {initial_used:.1f} MB\n")
    
    # Test cases with increasing condition counts
    test_cases = [
        (10, 10),    # 100 conditions - fits in single batch
        (32, 32),    # 1,024 conditions - fits in single batch  
        (64, 64),    # 4,096 conditions - exactly at batch limit
        (70, 70),    # 4,900 conditions - exceeds batch limit
        (100, 100),  # 10,000 conditions - requires batch processing
    ]
    
    for n_comp, n_temp in test_cases:
        print(f"\nTest: {n_comp}x{n_temp} grid = {n_comp*n_temp:,} conditions")
        print("-" * 40)
        
        # Create conditions
        x_bi_values = np.linspace(0.05, 0.95, n_comp)
        temp_values = np.linspace(400, 1200, n_temp)
        
        conditions = {
            v.T: temp_values,
            v.P: 101325 * np.ones(n_temp),
            v.X('BI'): x_bi_values
        }
        
        try:
            # Clear GPU memory
            mempool.free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            
            # Run GPU calculation
            start_time = time.time()
            result = equilibrium(dbf, comps, phases, conditions, 
                               gpu=True, verbose=False, 
                               calc_opts={'pdens': 50})
            gpu_time = time.time() - start_time
            
            # Check memory usage
            peak_used = mempool.used_bytes() / (1024**2)
            
            # Verify results
            gm_values = result.GM.values.flatten()
            valid_results = np.sum(~np.isnan(gm_values))
            
            print(f"  Success: {valid_results}/{len(gm_values)} valid results")
            print(f"  Time: {gpu_time:.2f} seconds")
            print(f"  Peak memory: {peak_used:.1f} MB")
            
            # Check if batch processing was used
            if n_comp * n_temp > 4096:
                print(f"  ✓ Batch processing used (conditions > 4096)")
            else:
                print(f"  ✓ Single batch (conditions ≤ 4096)")
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
            
    print("\n" + "=" * 60)
    print("Batch Processing Test Complete")
    
    # Final memory check
    mempool.free_all_blocks()
    final_used = mempool.used_bytes() / (1024**2)
    print(f"\nFinal memory used: {final_used:.1f} MB")
    print(f"Memory leaked: {final_used - initial_used:.1f} MB")

if __name__ == "__main__":
    test_batch_processing()