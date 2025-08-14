#!/usr/bin/env python
"""Compare CPU and GPU equilibrium matrices for single-phase system."""

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
    print("SINGLE-PHASE EQUILIBRIUM MATRIX COMPARISON")
    print("=" * 80)
    print(f"Phases: {phases}")
    print(f"Prescribed: X(AL)={conditions[v.X('AL')]:.3f}, X(CU)={conditions[v.X('CU')]:.3f}")
    print()
    
    # Run CPU with verbose to get matrix
    print("CPU Equilibrium Matrix:")
    cpu_result = equilibrium(dbf, comps, phases, conditions, gpu=False, verbose=True)
    
    print("\n" + "=" * 80)
    print("\nGPU Equilibrium Matrix:")
    gpu_result = equilibrium(dbf, comps, phases, conditions, gpu=True, verbose=True)

if __name__ == "__main__":
    main()