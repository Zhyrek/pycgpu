#!/usr/bin/env python
"""Analyze thread pattern for failing conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def analyze_failure_pattern():
    """Analyze which thread indices are failing."""
    
    print("Analysis of failing conditions in 32-thread batch:")
    print("=" * 60)
    
    # Map condition index to (X_BI, T) pairs
    conditions = []
    temps = [400, 500, 600, 700]
    x_bis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # The flattening order from the test
    idx = 0
    for t in temps:
        for x in x_bis:
            conditions.append((idx, x, t))
            idx += 1
    
    print(f"Total conditions: {len(conditions)}")
    print(f"\nCondition mapping:")
    for idx, x, t in conditions:
        print(f"  Thread {idx:2d}: X(BI)={x:.1f}, T={t:.0f}K")
    
    # Known failures from the output
    failed = [
        (10, 0.3, 500),  # Thread 10
        (17, 0.2, 600),  # Thread 17 (should be index 17)
    ]
    
    print(f"\nFailed conditions:")
    for idx, x, t in failed:
        modulo_7 = idx % 7
        modulo_8 = idx % 8
        modulo_32 = idx % 32
        print(f"  Thread {idx}: X(BI)={x}, T={t}K")
        print(f"    - idx % 7 = {modulo_7}")
        print(f"    - idx % 8 = {modulo_8}")
        print(f"    - idx % 32 = {modulo_32}")
        print(f"    - idx // 8 = {idx // 8} (temperature index)")
        print(f"    - idx % 8 = {idx % 8} (composition index)")
    
    # Check if there's a pattern
    print(f"\nPattern analysis:")
    print(f"  Both failures have (thread_idx % 7) == 3")
    print(f"  Thread 10 = 1*8 + 2 (T=500K, X_BI=0.3)")
    print(f"  Thread 17 = 2*8 + 1 (T=600K, X_BI=0.2)")
    
    # But wait, let me recalculate based on the actual output
    print(f"\nRechecking based on actual data:")
    # From the output file, line 12: X(BI)=0.3, T=500K failed
    # From the output file, line 19: X(BI)=0.2, T=600K failed
    
    # Count the actual indices
    actual_idx = 0
    for t_idx, t in enumerate(temps):
        for x_idx, x in enumerate(x_bis):
            if (x == 0.3 and t == 500) or (x == 0.2 and t == 600):
                print(f"  Actual index {actual_idx}: X(BI)={x}, T={t}K")
                print(f"    - idx % 7 = {actual_idx % 7}")
                print(f"    - idx % 8 = {actual_idx % 8}")
            actual_idx += 1
    
    print(f"\nMemory stride patterns:")
    print(f"  If using stride of 7, collisions would occur at:")
    for i in range(32):
        if i % 7 == 3:
            t_idx = i // 8
            x_idx = i % 8
            if t_idx < len(temps) and x_idx < len(x_bis):
                print(f"    Thread {i}: X(BI)={x_bis[x_idx]}, T={temps[t_idx]}K")

if __name__ == "__main__":
    analyze_failure_pattern()