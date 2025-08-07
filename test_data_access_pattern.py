#!/usr/bin/env python
"""Test if threads are accessing the correct SystemSpec data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def test_data_access():
    """Test if the correct data is being accessed by each thread."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Testing data access patterns...")
    print("=" * 60)
    
    # Create conditions that should have different X(BI) values
    x_bi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    temp_values = [400, 500, 600, 700]
    
    conditions = {
        v.X('BI'): x_bi_values,
        v.T: temp_values,
        v.P: 101325
    }
    
    print(f"Conditions grid: {len(temp_values)} temps x {len(x_bi_values)} compositions")
    print(f"Total conditions: {len(temp_values) * len(x_bi_values)}")
    print()
    
    # Map thread ID to expected condition
    print("Expected mapping (thread -> X(BI), T):")
    for tid in range(32):
        temp_idx = tid // 8
        comp_idx = tid % 8
        expected_x_bi = x_bi_values[comp_idx]
        expected_temp = temp_values[temp_idx]
        marker = " <-- FAILS" if tid in [10, 17] else ""
        print(f"  Thread {tid:2d}: X(BI)={expected_x_bi:.1f}, T={expected_temp}K{marker}")
    
    print()
    print("Analysis of failing threads:")
    print("  Thread 10: Should get X(BI)=0.3, T=500K")
    print("  Thread 17: Should get X(BI)=0.2, T=600K")
    print()
    
    # Now let's think about what could go wrong with stride-83 access
    print("Memory access analysis with stride=83:")
    print()
    
    # If there's an indexing bug related to the stride pattern
    for tid in [10, 17]:
        correct_offset = tid * 83
        
        # What if there's a bug where tid % 7 == 3 causes wrong indexing?
        # Perhaps the code is doing something like:
        # actual_offset = (tid // 7) * 7 * stride + (tid % 7) * stride
        # This would be wrong!
        
        wrong_offset_1 = (tid // 7) * 7 * 83 + (tid % 7) * 83
        wrong_offset_2 = ((tid // 7) * 83 * 7) + ((tid % 7) * 83)
        
        print(f"Thread {tid}:")
        print(f"  Correct offset: {correct_offset}")
        print(f"  If bug type 1: {wrong_offset_1} (same as correct)")
        print(f"  If bug type 2: {wrong_offset_2} (same as correct)")
        
        # What if threads are accidentally sharing data?
        # With stride 83 and threads 10 & 17 both having tid % 7 = 3
        # Could there be aliasing?
        
        # Check if the offset calculation could wrap around
        total_size = 32 * 83  # Total array size
        if correct_offset >= total_size:
            print(f"  WARNING: Offset {correct_offset} exceeds array size {total_size}!")
        
        # Check modulo patterns
        print(f"  tid % 7 = {tid % 7}")
        print(f"  offset % 64 = {correct_offset % 64} (cache line alignment)")
        print(f"  offset % 128 = {correct_offset % 128}")
        print()
    
    # The real issue might be that the solver diverges for these specific conditions
    # when run in parallel due to numerical instability from cache conflicts
    print("Hypothesis: The stride-7 pattern causes cache conflicts that lead to:")
    print("1. Slower memory access for threads 10 & 17")
    print("2. This causes these threads to read stale/partially updated data")
    print("3. Or causes race conditions in the iterative solver")
    print("4. Leading to solver divergence (not convergence)")
    print()
    
    # Let's check what the actual error is
    print("From the test output:")
    print("  Thread 10: Converged: NO, Delta GM = 331.472032")
    print("  Thread 17: Converged: NO, Delta GM = -1227.369024")
    print()
    print("These threads didn't converge! It's not wrong data access,")
    print("it's that the solver fails to converge for these specific threads.")
    print()
    print("This suggests the issue is:")
    print("- Memory access conflicts cause the solver to diverge")
    print("- OR these threads are reading partially written data from other threads")
    print("- OR there's a race condition in shared memory access")

if __name__ == "__main__":
    test_data_access()