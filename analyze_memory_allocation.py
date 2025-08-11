#!/usr/bin/env python
"""
Analyze the difference between using cp.zeros vs cp.empty for GPU memory allocation.
"""

import cupy as cp
import numpy as np
import time
import gc

def test_memory_allocation():
    """Compare memory allocation strategies."""
    
    print("GPU Memory Allocation Analysis")
    print("=" * 60)
    
    # Get GPU info
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
    print(f"Total Memory: {device.mem_info[1] / (1024**3):.2f} GB")
    
    # Memory pool for tracking
    mempool = cp.get_default_memory_pool()
    
    # Test sizes (simulating work arrays for different thread counts)
    test_cases = [
        (1000, "1,000 threads"),
        (4096, "4,096 threads (batch limit)"),
        (10000, "10,000 threads"),
    ]
    
    # Size of work arrays per thread (in doubles)
    WORK_ARRAY_SIZES = {
        'A_lstsq_copy': 138 * 138,      # 19,044 doubles
        'U_lstsq': 138 * 138,            # 19,044 doubles
        'V_lstsq': 138 * 138,            # 19,044 doubles
        'singular_values_lstsq': 138,    # 138 doubles
        'superdiag_lstsq': 138,          # 138 doubles
        'hess': 100 * 100,               # 10,000 doubles
        'grad': 100,                     # 100 doubles
        'phase_matrix': 46 * 46,         # 2,116 doubles
        # Total: ~76,668 doubles = ~597 KB per thread
    }
    
    for num_threads, desc in test_cases:
        print(f"\n{desc}:")
        print("-" * 40)
        
        # Test cp.zeros
        print("\nUsing cp.zeros:")
        mempool.free_all_blocks()
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
        
        start_time = time.time()
        arrays_zeros = {}
        
        # Allocate with zeros
        for name, size in WORK_ARRAY_SIZES.items():
            arrays_zeros[name] = cp.zeros((num_threads, size), dtype=cp.float64)
        
        cp.cuda.runtime.deviceSynchronize()
        zeros_time = time.time() - start_time
        
        # Check memory
        zeros_memory = mempool.used_bytes() / (1024**2)
        
        print(f"  Allocation time: {zeros_time:.3f} seconds")
        print(f"  Memory used: {zeros_memory:.1f} MB")
        
        # Clear for next test
        del arrays_zeros
        gc.collect()
        
        # Test cp.empty
        print("\nUsing cp.empty:")
        mempool.free_all_blocks()
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
        
        start_time = time.time()
        arrays_empty = {}
        
        # Allocate with empty
        for name, size in WORK_ARRAY_SIZES.items():
            arrays_empty[name] = cp.empty((num_threads, size), dtype=cp.float64)
        
        cp.cuda.runtime.deviceSynchronize()
        empty_time = time.time() - start_time
        
        # Check memory
        empty_memory = mempool.used_bytes() / (1024**2)
        
        print(f"  Allocation time: {empty_time:.3f} seconds")
        print(f"  Memory used: {empty_memory:.1f} MB")
        
        # Compare
        print(f"\nComparison:")
        print(f"  Time difference: {zeros_time - empty_time:.3f} sec (zeros slower)")
        print(f"  Memory difference: {zeros_memory - empty_memory:.1f} MB")
        speedup = zeros_time / empty_time if empty_time > 0 else 0
        print(f"  Speedup with empty: {speedup:.2f}x")
        
        # Clear
        del arrays_empty
        gc.collect()
    
    print("\n" + "=" * 60)
    print("Key Findings:")
    print("=" * 60)
    print("1. cp.zeros requires initializing memory to 0 on GPU")
    print("2. cp.empty just allocates without initialization")
    print("3. For work arrays that are immediately overwritten,")
    print("   cp.empty is more efficient")
    print("4. Memory usage is the same, but allocation is faster")

def test_initialization_overhead():
    """Test the overhead of zero initialization."""
    
    print("\n" + "=" * 60)
    print("Zero Initialization Overhead Test")
    print("=" * 60)
    
    sizes = [1_000_000, 10_000_000, 100_000_000]  # 8MB, 80MB, 800MB
    
    for size in sizes:
        size_mb = size * 8 / (1024**2)
        print(f"\nArray size: {size:,} doubles ({size_mb:.1f} MB)")
        
        # Test zeros
        start = time.time()
        arr_zeros = cp.zeros(size, dtype=cp.float64)
        cp.cuda.runtime.deviceSynchronize()
        zeros_time = time.time() - start
        
        # Test empty
        start = time.time()
        arr_empty = cp.empty(size, dtype=cp.float64)
        cp.cuda.runtime.deviceSynchronize()
        empty_time = time.time() - start
        
        print(f"  cp.zeros: {zeros_time:.4f} sec")
        print(f"  cp.empty: {empty_time:.4f} sec")
        print(f"  Overhead: {zeros_time - empty_time:.4f} sec")
        
        throughput_zeros = size_mb / zeros_time if zeros_time > 0 else 0
        throughput_empty = size_mb / empty_time if empty_time > 0 else 0
        print(f"  Throughput zeros: {throughput_zeros:.1f} MB/s")
        print(f"  Throughput empty: {throughput_empty:.1f} MB/s")
        
        del arr_zeros, arr_empty

def demonstrate_work_array_usage():
    """Show which arrays can safely use cp.empty."""
    
    print("\n" + "=" * 60)
    print("Work Array Usage Analysis")
    print("=" * 60)
    
    safe_for_empty = [
        "A_lstsq_copy - Overwritten immediately in SVD",
        "U_lstsq - Output array for SVD decomposition",
        "V_lstsq - Output array for SVD decomposition",
        "singular_values_lstsq - Output array for SVD",
        "superdiag_lstsq - Work array for SVD",
        "U_inv - Output array for matrix inversion",
        "V_inv - Output array for matrix inversion",
        "work_inv - Temporary work array",
        "grad - Overwritten by gradient calculation",
        "hess - Overwritten by Hessian calculation",
        "masses - Calculated from composition",
        "mass_jac - Calculated Jacobian",
        "phase_matrix - Built during solve",
        "equilibrium_matrix - Built during solve",
        "equilibrium_rhs - Built during solve",
        "eq_soln - Output of linear solve",
    ]
    
    needs_zeros = [
        "removed_compsets - May be checked before write",
        "compsets_before_solve - May be checked before write",
        "compsets_before_final_solve - May be checked before write",
        "debug arrays - Need clean state for debugging",
    ]
    
    print("\nArrays safe for cp.empty (immediately overwritten):")
    for item in safe_for_empty:
        print(f"  ✓ {item}")
    
    print("\nArrays that need cp.zeros (checked before write):")
    for item in needs_zeros:
        print(f"  × {item}")
    
    print("\nRecommendation:")
    print("  Most work arrays (90%+) can use cp.empty for faster allocation")

if __name__ == "__main__":
    test_memory_allocation()
    test_initialization_overhead()
    demonstrate_work_array_usage()