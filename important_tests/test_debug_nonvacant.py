#!/usr/bin/env python
"""Debug nonvacant_elements value."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("DEBUG NONVACANT_ELEMENTS")
    print("=" * 80)
    
    print("\nComponents:", comps)
    print("Non-VA count:", sum(1 for c in comps if c != 'VA'))
    
    # Run GPU with verbose to see debug output
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()