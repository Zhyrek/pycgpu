#!/usr/bin/env python
"""Compare CPU and GPU matrices at each iteration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYCALPHAD_DEBUG_MODE'] = '1'

import numpy as np
from pycalphad import Database, equilibrium, variables as v
import warnings
warnings.filterwarnings("ignore")

def main():
    dbf = Database('Al-Cu-Fe.tdb')
    comps = ['AL', 'CU', 'FE', 'VA']
    phases = ['LIQUID']
    
    conditions = {
        v.X('AL'): 0.25,
        v.X('CU'): 0.25,
        v.T: 2000,
        v.P: 101325
    }
    
    print("=" * 80)
    print("TESTING SINGLE-PHASE SYSTEM")
    print("=" * 80)
    print(f"Phases: {phases}")
    print(f"Prescribed: X(AL)={conditions[v.X('AL')]:.3f}, X(CU)={conditions[v.X('CU')]:.3f}")
    print("\nRunning calculations and outputting matrices at each iteration...")
    print("\n(Look for [EQUILIBRIUM_MATRIX_OUTPUT] lines in the output)")
    print()
    
    # Run both to generate matrix output
    equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()