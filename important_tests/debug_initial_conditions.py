#!/usr/bin/env python
"""Debug initial conditions for CPU vs GPU."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pycalphad import Database, equilibrium, variables as v
from pycalphad.core.minimizer import set_debug_mode
import warnings
warnings.filterwarnings("ignore")

# Enable debug mode
set_debug_mode(True)

def main():
    dbf = Database('../Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['BCC_B2', 'AL5FE2']
    
    conditions = {
        v.X('AL'): 0.60,
        v.X('CU'): 0.10,
        v.T: 600,
        v.P: 101325
    }
    
    print("=" * 80)
    print("INITIAL CONDITIONS DEBUG")
    print("=" * 80)
    
    print("\nRunning CPU calculation with debug output...")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\n" + "=" * 80)
    print("\nRunning GPU calculation with debug output...")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()