#!/usr/bin/env python
"""Check if padding is being applied correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.utils import filter_phases

# Test the padding directly
from pycalphad.gpu.gpu_systemspec_flat import create_flat_system_specification, apply_safe_padding

def test_padding():
    """Test if padding is working."""
    
    print("Testing padding function...")
    print("=" * 60)
    
    # Create a dummy spec of size 83 (the problematic size)
    spec = np.ones(83, dtype=np.float64)
    
    print(f"Original size: {len(spec)} doubles")
    
    # Apply padding
    padded = apply_safe_padding(spec, verbose=True)
    
    print(f"Padded size: {len(padded)} doubles")
    print(f"Padding added: {len(padded) - len(spec)} doubles")
    
    # Check the pattern
    print(f"\nPattern analysis:")
    print(f"  Original size % 7 = {len(spec) % 7}")
    print(f"  Padded size % 7 = {len(padded) % 7}")
    print(f"  Original would cause 7-stride conflicts: {len(spec) % 7 == 6}")
    print(f"  Padded avoids conflicts: {len(padded) % 7 != 6 and len(padded) % 7 != 0}")
    
    # Now test with actual equilibrium calculation
    print("\n" + "=" * 60)
    print("Testing with actual calculation (verbose mode)...")
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = filter_phases(dbf, comps)
    
    # Small test with 4 conditions to see debug output
    conditions = {
        v.X('BI'): [0.3, 0.2],  # The failing conditions
        v.T: [500, 600],
        v.P: 101325
    }
    
    print(f"\nRunning GPU calculation with 2x2=4 conditions...")
    result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    print(f"\nCalculation completed")

if __name__ == "__main__":
    test_padding()