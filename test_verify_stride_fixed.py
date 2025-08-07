#!/usr/bin/env python
"""Verify the actual SystemSpec stride value being used - fixed version."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def test_stride():
    """Test the actual stride value."""
    
    print("Testing SystemSpec stride value...")
    print("=" * 60)
    
    # Test with 32 conditions (the failing case)
    print(f"Running with 4x8 = 32 conditions")
    print(f"Expected threads with conflicts (tid % 7 == 3): [3, 10, 17, 24, 31]")
    print(f"Actual failing threads from test: 10 and 17")
    print()
    
    # Based on the code analysis, the SystemSpec has these sizes:
    MAX_COMPONENTS = 3
    MAX_STATEVARS = 3
    MAX_PHASES = 6
    MAX_FIXED_MOLE = 3
    MAX_DOF_PER_PHASE = 3  # Added this
    
    # From gpu_systemspec_flat.py, the flat array contains:
    # - 50 scalar fields (global_spec_np)
    # - initial_chemical_potentials: MAX_COMPONENTS = 3
    # - prescribed_mole_fraction_coefficients: MAX_FIXED_MOLE * MAX_COMPONENTS = 9
    # - prescribed_mole_fraction_rhs: MAX_FIXED_MOLE = 3
    # - free_chemical_potential_indices: MAX_COMPONENTS = 3
    # - free_statevar_indices: MAX_STATEVARS = 3
    # - fixed_chemical_potential_indices: MAX_COMPONENTS = 3
    # - fixed_statevar_indices: MAX_STATEVARS = 3
    # - fixed_stable_compset_indices: MAX_PHASES = 6
    
    spec_size = 50 + 3 + 9 + 3 + 3 + 3 + 3 + 3 + 6
    
    print(f"SystemSpec size: {spec_size} doubles")
    print(f"SystemSpec size % 7 = {spec_size % 7}")
    print()
    
    # 83 % 7 = 6, so threads separated by 7 will have addresses differing by 7*83 = 581
    print(f"Analysis:")
    print(f"  Stride = {spec_size} doubles")
    print(f"  {spec_size} % 7 = {spec_size % 7}")
    print(f"  Thread separation = 7 * {spec_size} = {7 * spec_size} doubles")
    print(f"  {7 * spec_size} % 7 = {(7 * spec_size) % 7} (perfect conflict!)")
    print()
    print(f"This means threads 3, 10, 17, 24, 31 all access memory at:")
    for tid in [3, 10, 17, 24, 31]:
        offset = tid * spec_size
        print(f"  Thread {tid:2d}: offset {offset:4d} = {3*spec_size} + {(tid-3)*spec_size}")
    print()
    print(f"The stride of {7*spec_size} doubles between conflicting threads creates cache conflicts")
    print(f"because they all map to the same cache set modulo the cache line size.")
    print()
    print("SOLUTION: Pad the SystemSpec array to avoid stride-7 conflicts.")
    print("We need to pad from 83 to a number that doesn't create conflicts.")
    print()
    
    # Test different padding sizes
    print("Testing padding options:")
    for pad_size in [84, 85, 86, 87, 88, 89, 90, 91, 96]:
        conflicts = []
        for tid in range(32):
            if tid % 7 == 3:
                conflicts.append(tid)
        
        # Check if the stride between conflicting threads is problematic
        if len(conflicts) >= 2:
            tid1, tid2 = conflicts[0], conflicts[1]
            diff = (tid2 - tid1) * pad_size
            problematic = (diff % 7 == 0) or (diff % 64 == 0)  # Cache line is typically 64 bytes = 8 doubles
            
            status = "❌ CONFLICT" if problematic else "✓ OK"
            print(f"  Pad to {pad_size}: {pad_size % 7} mod 7, diff={diff}, {status}")

if __name__ == "__main__":
    test_stride()