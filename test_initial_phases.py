#!/usr/bin/env python
"""Test to see initial phase configuration for GPU vs CPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def test_initial_phases():
    """Check initial phase configuration."""
    
    dbf = Database('important_tests/AuBi-07Wan.tdb')
    comps = ['AU', 'BI', 'VA']
    phases = ['AU2BI_C15', 'BCC_A2', 'FCC_A1', 'HCP_A3', 'LIQUID', 'RHOMBOHEDRAL_A7']
    
    conditions = {
        v.X('BI'): 0.1,
        v.T: 400,
        v.P: 101325
    }
    
    print("="*70)
    print("INITIAL PHASE CONFIGURATION TEST")
    print("="*70)
    
    # Test with verbose GPU to see initial configuration
    print("\n--- GPU VERBOSE OUTPUT ---")
    print("Look for:")
    print("  1. Starting point phases (should be FCC_A1 + AU2BI_C15)")
    print("  2. Initial compset configuration")
    print("  3. Why 4 phases are active at iteration 0")
    print("\n")
    
    # Set environment variable for extra debug
    import os
    os.environ['GPU_DEBUG_INIT'] = '1'
    
    result_gpu = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)
    
    print("\n" + "="*70)
    print("KEY OBSERVATION")
    print("="*70)
    print("The GPU starts with 4 active phases (compsets) instead of 2.")
    print("This suggests the GPU is not properly using the starting point")
    print("and instead initializes all phases or a default set.")

if __name__ == "__main__":
    test_initial_phases()