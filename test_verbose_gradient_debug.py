#!/usr/bin/env python
"""Test with VERBOSE_DEBUG to see actual gradient values."""

from pycalphad import Database, equilibrium, variables as v
import numpy as np
import io
import sys
import re
import os

def test_with_verbose_debug():
    """Test with VERBOSE_DEBUG compilation to see actual gradient mapping."""
    
    print("TESTING AL-CU-FE TERNARY WITH VERBOSE_DEBUG")
    print("="*60)
    
    try:
        tdb = Database('Al-Cu-Fe.tdb')
        
        # Set environment variable to enable VERBOSE_DEBUG
        os.environ['PYCALPHAD_GPU_DEBUG'] = '1'
        
        # Test single ternary condition
        eq = equilibrium(tdb, ['AL', 'CU', 'FE', 'VA'], ['LIQUID'], {
            v.T: 1200, v.P: 101325, v.X('CU'): 0.3, v.X('FE'): 0.2
        }, gpu=True, verbose=True)
        
        print("✓ Equilibrium calculation completed")
        print(f"Final GM: {eq.GM.values[0]:.6f} J/mol")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_verbose_debug()