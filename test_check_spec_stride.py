#!/usr/bin/env python
"""Check the SystemSpecification stride calculation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

def check_spec_stride():
    """Check the SystemSpecification stride for different batch sizes."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    print("Testing SystemSpecification stride patterns...")
    print("=" * 60)
    
    # Test different batch sizes
    test_cases = [
        # (num_temps, num_comps, description)
        (1, 1, "Single condition"),
        (1, 2, "2 conditions (1x2)"),
        (2, 2, "4 conditions (2x2)"),
        (3, 3, "9 conditions (3x3)"),
        (4, 8, "32 conditions (4x8) - FAILS"),
        (5, 8, "40 conditions (5x8)"),
    ]
    
    for num_temps, num_comps, desc in test_cases:
        print(f"\nTest: {desc}")
        print(f"  Grid: {num_temps} temperatures x {num_comps} compositions")
        
        total_conditions = num_temps * num_comps
        print(f"  Total conditions: {total_conditions}")
        
        # Simulate the stride calculation
        # From gpu_systemspec_flat.py, the SystemSpec has many fields
        # Let's calculate the expected size
        
        MAX_COMPONENTS = 3
        MAX_STATEVARS = 3
        MAX_PHASES = 6
        MAX_FIXED_MOLE = 3
        
        # From create_flat_system_specification in gpu_systemspec_flat.py
        # The spec contains scalars + arrays
        scalar_fields = 50  # From global_spec_np = np.zeros(50, dtype=np.float64)
        
        # Arrays:
        # initial_chemical_potentials: MAX_COMPONENTS
        # prescribed_mole_fraction_coefficients: MAX_FIXED_MOLE * MAX_COMPONENTS
        # prescribed_mole_fraction_rhs: MAX_FIXED_MOLE
        # free_chemical_potential_indices: MAX_COMPONENTS (stored as doubles)
        # free_statevar_indices: MAX_STATEVARS (stored as doubles)
        # fixed_chemical_potential_indices: MAX_COMPONENTS (stored as doubles)
        # fixed_statevar_indices: MAX_STATEVARS (stored as doubles)
        # fixed_stable_compset_indices: MAX_PHASES (stored as doubles)
        
        array_size = (
            MAX_COMPONENTS +  # initial_chemical_potentials
            MAX_FIXED_MOLE * MAX_COMPONENTS +  # prescribed_mole_fraction_coefficients
            MAX_FIXED_MOLE +  # prescribed_mole_fraction_rhs
            MAX_COMPONENTS +  # free_chemical_potential_indices
            MAX_STATEVARS +  # free_statevar_indices
            MAX_COMPONENTS +  # fixed_chemical_potential_indices
            MAX_STATEVARS +  # fixed_statevar_indices
            MAX_PHASES  # fixed_stable_compset_indices
        )
        
        spec_size = scalar_fields + array_size
        print(f"  Expected SystemSpec size: {spec_size} doubles")
        
        # Total array size
        total_array_size = total_conditions * spec_size
        
        # Calculate stride (should equal spec_size)
        stride = total_array_size // total_conditions
        print(f"  Calculated stride: {stride} doubles")
        
        # Check for problematic modulo patterns
        print(f"  Checking thread access patterns:")
        failures = []
        for tid in range(total_conditions):
            # Check various modulo patterns
            if tid % 7 == 3:
                failures.append(tid)
        
        if failures:
            print(f"    ⚠️  Threads with (tid % 7 == 3): {failures}")
            print(f"    These threads may experience memory conflicts!")
        else:
            print(f"    ✓  No threads match problematic pattern (tid % 7 == 3)")
        
        # Calculate memory addresses for problematic threads
        if failures and len(failures) >= 2:
            tid1, tid2 = failures[0], failures[1]
            addr1 = tid1 * stride
            addr2 = tid2 * stride
            diff = addr2 - addr1
            print(f"    Memory layout:")
            print(f"      Thread {tid1} starts at offset {addr1}")
            print(f"      Thread {tid2} starts at offset {addr2}")
            print(f"      Difference: {diff} doubles = {diff * 8} bytes")
            print(f"      Difference % 7 = {diff % 7}")
            
            # Check if stride itself is problematic
            if stride % 7 == 0:
                print(f"    ⚠️  Stride {stride} is divisible by 7 - potential cache conflict!")
            
if __name__ == "__main__":
    check_spec_stride()